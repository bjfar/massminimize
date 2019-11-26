"""Core optimization routines and tools"""

import numpy as np
import tensorflow as tf
import datetime

# Better version of 'partial'. From 'funcy' library: https://github.com/Suor/funcy
def func_partial(func, *args, **kwargs):
    """A functools.partial alternative, which returns a real function.
       Can be used to construct methods."""
    return lambda *a, **kw: func(*(args + a), **dict(kwargs, **kw))

# Nested list/tuple flattener. From https://stackoverflow.com/a/10824420/1447953
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def optimize(parslist,lossf,step,tol=1e-2,grad_tol=1e-4,conv_from_trans_pars=False):

    opt = tf.optimizers.SGD(step)
    i=0
    max_grad=1e99
    max_delta_grad = 1e99
    prev_grads = [None for p in flatten(parslist)]
    prev_parslist = [None for p in flatten(parslist)]
    mask = None
    start = datetime.datetime.now()
    all_converged = False
    prev_Nconverged = 0 
    Nconverged_same_count = 0

    while not all_converged or i<10:
        with tf.GradientTape(persistent=True) as tape:
            total_loss, pars_real, extra = lossf(parslist)

        # Flatten input TensorFlow variables list. User can structure it however they
        # want for their convenience in their loss function, but here we just want a 
        # flat list of variables.
        flatparslist = list(flatten(parslist))

        gradients_in = tape.gradient(total_loss, flatparslist)
        #print("gradients_in:", gradients_in)
        #print("parslist:", parslist)
        opt.apply_gradients(zip(gradients_in, flatparslist))

        # Now we need the gradients w.r.t. the non-weirdly scaled parameters, for convergence calculations
        #all_gradients = [tape.gradient(total_loss, pars) for pars in flatten(pars_real)]
        if conv_from_trans_pars:
            flatpars = list(flatten(pars_real))
            all_gradients = tape.gradient(total_loss, flatpars)
        else:
            # No need to re-do gradient backprop
            flatpars = flatparslist
            all_gradients = gradients_in

        # Need separate convergence criteria for each separate minimisation problem
        # (even though we are doing them all at once)
        N_converged = 0
        N_tot = 0
        tot_max_grad = 0
        tot_max_delta_grad = 0
        tot_max_delta_f = 0
        for j,(grads,pars,pgrads,ppars) in enumerate(zip(all_gradients,flatpars,prev_grads,prev_parslist)):
            grad_norms = grads
            if pgrads is None:
                delta_grads = grad_norms*0 + 1e99
                delta_f = (grad_norms*0 + 1e99).numpy()
            else:
                delta_grads = tf.abs(grad_norms - pgrads)
                #delta_mus = np.sqrt(tf.reduce_sum(tf.square(tf.reshape(pars.numpy() - ppars,(pars.shape[0],-1))),axis=1))
                delta_mus = np.abs(pars.numpy() - ppars)
                #print("delta_mus:",delta_mus)
                delta2_approx = delta_grads.numpy() / delta_mus # Super dodgy approximation of second derivative
                dist_to_min = grad_norms.numpy() / delta2_approx # Dodgy estimate of distance to minimum
                delta_f = delta_grads.numpy() * dist_to_min # Dodgy estimate of remaining error on loss function
            prev_parslist[j] = pars.numpy()
            prev_grads[j]    = grad_norms
            max_delta_grad = tf.reduce_max(delta_grads)
            avg_grad = tf.reduce_mean(grad_norms)
            max_grad = tf.reduce_max(grad_norms)
            converged = (grad_norms.numpy()<tol) | (delta_grads.numpy()<grad_tol)
            #print("delta_f:", delta_f)
            #print("delta_grads:", delta_grads)
            #print("grad converged:", np.sum(grad_norms.numpy()<tol))
            #print("delta grad converged:", np.sum(delta_grads.numpy()<grad_tol)) 
            #print("func converged:", np.sum(delta_f<tol))
            #converged = (delta_f<tol) | (delta_grads.numpy()<grad_tol) 
            #print("j:",j," converged: ",converged)
            N_tot += np.prod(converged.shape)
 
            # Selection of trials that have *not* converged
            mask1 = tf.greater_equal(grad_norms,  tol      * tf.ones_like(grad_norms))
            mask2 = tf.greater_equal(delta_grads, grad_tol * tf.ones_like(delta_grads))
            mask = tf.logical_and(mask1, mask2)
    
            #tot_max_grad       = np.max([tot_max_grad      ,max_grad])
            #tot_max_delta_grad = np.max([tot_max_delta_grad,max_delta_grad])
            #tot_max_delta_f    = np.max([tot_max_delta_f   ,np.max(delta_f)])
            tot_max_grad       = max_grad
            tot_max_delta_grad = max_delta_grad
            tot_max_delta_f    = np.max(delta_f)
     
            N_converged += np.sum(converged) 
        #print("i: {0}, max_grad: {1}, max_delta_grad: {2}, N_converged: {3}".format(i,max_grad,max_delta_grad,N_converged))
        print("i: {0}, max_grad: {1}, max_delta_grad: {2}, max_delta_f: {3}, N_converged: {4}/{5}".format(i,tot_max_grad,tot_max_delta_grad,tot_max_delta_f,N_converged,N_tot))
    
        if N_converged == prev_Nconverged: 
            Nconverged_same_count += 1
        else:
            Nconverged_same_count = 0

        if i>30 and Nconverged_same_count>10:
            print("WARNING! {0} problems failed to converged after {1} iterations. Stopping anyway.".format(N_tot - N_converged,i))
            all_converged=True

        if i>1000:
            print("WARNING! {0} problems failed to converged after {1} iterations. Stopping anyway.".format(N_tot - N_converged,i))
            all_converged=True
 
        prev_Nconverged = N_converged    

        if N_converged == N_tot: all_converged = True
        i+=1
    elapsed = datetime.datetime.now() - start
    
    print("Time elapsed (s): {0}".format(elapsed.total_seconds()))
    print("Number of iterations performed: {0}".format(i))
    if i==2:
        print("WARNING! Iteration 'converged' immediately! Please check the convergence criteria and try again")

    return pars_real, extra

