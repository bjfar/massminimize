"""Basic optimisation in TensorFlow

   In this test I am attemping to optimise for multiple
   "signal hypotheses", as well as over the pseudodata
   trials.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

import massminimize as mm

# Profiling tools
#import cProfile, pstats, sys
#pr = cProfile.Profile()

# Change plotting resolution due to my high DPI monitor
#matplotlib.rcParams['figure.dpi'] = 200

Npars = 4
N = int(1e4)

# Need to construct these shapes to match the event_shape, batch_shape, sample_shape 
# semantics of tensorflow_probability.

# Signal hypotheses (same for all components)
# (we will have the same null hypothesis for all of them, s=0)
s_in = tf.constant([i for i in range(10)],dtype=float)
#s_in = tf.constant([5],dtype=float)
s_in2 = tf.expand_dims(s_in, 0)
s = tf.broadcast_to(s_in2,shape=(N, len(s_in)))

b = tf.expand_dims(tf.constant(50,dtype=float),0)

# Nuisance parameters (independent Gaussians)
zero = tf.expand_dims(tf.constant(0,dtype=float),0)
nuis0 = [tfd.Normal(loc = zero, scale = 1) for i in range(Npars)]

# Bunch of independent Poisson distributions that we want to combine
poises0  = [tfd.Poisson(rate = b) for i in range(Npars)]
poises0s = [tfd.Poisson(rate = s_in + b) for i in range(Npars)]

# Construct joint distributions
joint0 = tfd.JointDistributionSequential(poises0+nuis0)
joint0s = tfd.JointDistributionSequential(poises0s+nuis0)

# Generate background-only pseudodata to be fitted
samples0 = joint0.sample(N)

# Generate signal+background pseudodata to be fitted
samples0s = joint0s.sample(N)

# We want the sample shapes to dimensionally match the versions of
# the distributions that have free parameters:
# [sample_shape, batch_shape, event_shape]

print("[sample_shape, batch_shape, event_shape]")
print("joint0.batch_shape:",joint0.batch_shape[0])
print("joint0.event_shape:",joint0.event_shape[0])
print("samples0.shape:", [k.shape for k in samples0][0])
print("joint0s.batch_shape:",joint0s.batch_shape[0])
print("joint0s.event_shape:",joint0s.event_shape[0])
print("samples0s.shape:", [k.shape for k in samples0s][0])

# Generate Asimov datasets (no need to fit due to Asimov dataset properties)
# Counts equal to expected values, nuisance pars equal to 0
# Need to get the tensor shapes right so that they evaluate one sample for each signal hypothesis,
# rather than broadcasting or some such.
samplesAsb = [b+s_in for i in range(Npars)] + [tf.broadcast_to(tf.constant(0,dtype=float),shape=(len(s_in),)) for i in range(Npars)]
samplesAb  = [tf.constant(b,dtype=float) for i in range(Npars)] + [tf.constant(0,dtype=float) for i in range(Npars)]

#print([k.shape for k in samples0])
#print([k.shape for k in samplesA])

# Evaluate distributions for Asimov dataset 
qsbAsb = -2*(joint0s.log_prob(samplesAsb))
qbAsb  = -2*(joint0.log_prob(samplesAsb)) 
qAsb = qsbAsb - qbAsb

qsbAb = -2*(joint0s.log_prob(samplesAb))
qbAb  = -2*(joint0.log_prob(samplesAb)) 
qAb = qsbAb - qbAb

#q0 =-2*joint0.log_prob(samples0)
#print("q0:", q0)

# Parameters to optimize for each trial (nuisance parameters)
# Separate optimisation for s+b hypothesis and b-only hypothesis
# Single optimisation for b-hypothesis, optimisation for every s+b hypothesis
thetasb = [tf.Variable(np.zeros(s.shape, dtype='float32'), name='theta_sb{0}'.format(i)) for i in range(Npars)]
thetab = [tf.Variable(np.zeros((N,1), dtype='float32'), name='theta_b{0}'.format(i)) for i in range(Npars)] # Make sure matches thetasb dimensions for consistent broadcasting!

# Need the same for fit to signal+background pseudodata. This time seperate fits needed for Lb component because pseudodata is different
thetasb_s = [tf.Variable(np.zeros(s.shape, dtype='float32'), name='theta_sb_s{0}'.format(i)) for i in range(Npars)]
thetab_s = [tf.Variable(np.zeros(s.shape, dtype='float32'), name='theta_b_s{0}'.format(i)) for i in range(Npars)]

# Need to group parameters together into their common minimisation problems, so that we can track
# convergence
parslist_b  = [thetasb,thetab]
parslist_sb = [thetasb_s,thetab_s]

N_tot = len(s_in)*3*N + N # Total number of optimisations occurring

def lossf(pars,data):
    thetasb, thetab = pars
    nuis_sb  = [tfd.Normal(loc = thetasb[i], scale = 1) for i in range(Npars)]
    poises_sb= [tfd.Poisson(rate = s + b + thetasb[i]) for i in range(Npars)]
    joint_sb = tfd.JointDistributionSequential(poises_sb+nuis_sb)

    nuis_b  = [tfd.Normal(loc = thetab[i], scale = 1) for i in range(Npars)]
    poises_b= [tfd.Poisson(rate = b + thetab[i]) for i in range(Npars)]
    joint_b = tfd.JointDistributionSequential(poises_b+nuis_b)

    # Tensor shape matching debugging
    #print("[sample_shape, batch_shape, event_shape]")
    #print("joint_sb.batch_shape:",joint_sb.batch_shape[0])
    #print("joint_sb.event_shape:",joint_sb.event_shape[0])
    #print("samples shapes:", [k.shape for k in samples0][0])

    # The broadcasting works like this:

    # 1. Define n = len(batch_shape) + len(event_shape). (For scalar distributions, len(event_shape)=0.)
    # 2. If the input tensor t has fewer than n dimensions, pad its shape by adding dimensions of size 1 on the left until it has exactly n dimensions. Call the resulting tensor t'.
    # 3. Broadcast the n rightmost dimensions of t' against the [batch_shape, event_shape] of the distribution you're computing a log_prob for. In more detail: for the dimensions where t' already matches the distribution, do nothing, and for the dimensions where t' has a singleton, replicate that singleton the appropriate number of times. Any other situation is an error. (For scalar distributions, we only broadcast against batch_shape, since event_shape = [].)
    # 4. Now we're finally able to compute the log_prob. The resulting tensor will have shape [sample_shape, batch_shape], where sample_shape is defined to be any dimensions of t or t' to the left of the n-rightmost dimensions: sample_shape = shape(t)[:-n].

    # We have, e.g.
    # joint_sb.batch_shape: (10000, 5)
    # joint_sb.event_shape: ()
    # and we want to compute (10000, 5) log probabilities, broadcasting
    # 10000 samples over the "5" dimension.
    # So for that, according to the above rules, the input sample tensor shape
    # should be: (10000, 1)
    # And the resulting log-probability tensor should have shape
    # (10000, 5)

    qsb = -2*(joint_sb.log_prob(data))
    qb  = -2*(joint_b.log_prob(data))

    #print("qsb.shape:", qsb.shape)
    #print("qsb_s.shape:", qsb_s.shape)

    total_loss = tf.math.reduce_sum(qsb) + tf.math.reduce_sum(qb)

    # First return: total loss function value
    # Second return: 'true' parameter values (for convergence calculations)
    # Third return: extra variables whose final values you want to know at the end of the optimisation
    return total_loss, (thetasb,thetab), (qsb,qb)

# Optimise w.r.t. background-only samples
#pr.enable()
pars, (qsb,qb) = mm.optimize(parslist_b,mm.func_partial(lossf,data=samples0),step=0.01,tol=0.1,grad_tol=1e-4)
#pars, (qsb,qb) = mm.optimize(parslist_b,mm.func_partial(lossf,data=samples0),step=0.01,tol=0.5,grad_tol=1e-3)
#pr.disable()

#ps = pstats.Stats(pr, stream=sys.stdout)
#ps.sort_stats('cumulative').print_stats()
#quit()

# Optimise w.r.t. signal samples
pars_s, (qsb_s,qb_s) = mm.optimize(parslist_sb,mm.func_partial(lossf,data=samples0s),step=0.01,tol=0.1,grad_tol=1e-4)

#print("parameters:", mus)
#print("loss:", total_loss)
#print("qsb:", qsb)
#print("qb:", qb)
q = qsb - qb
#print("q:", q)

#print("qsb_s:", qsb_s)
#print("qb_s:", qb_s)
q_s = qsb_s - qb_s
#print("q_s:", q_s)


nplots = len(s_in)
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax2.set(yscale="log")
for i in range(nplots):
    sns.distplot(q[:,i], kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.distplot(q_s[:,i], kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))
     #qx = np.linspace(np.min(q),np.max(q),1000)
    #qy = np.exp(tfd.Chi2(df=Npars).log_prob(qx))
    #sns.lineplot(qx,qy)
    sns.distplot(q[:,i], kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.distplot(q_s[:,i], kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
     #qx = np.linspace(np.min(q),np.max(q),1000)
    #qy = np.exp(tfd.Chi2(df=Npars).log_prob(qx))
    #sns.lineplot(qx,qy)
    #ax.set_ylim(1./N,1.01*np.max(q)]))
ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)

fig = plt.figure(figsize=(12,4*nplots))
for i in range(nplots):
    ax1 = fig.add_subplot(nplots,2,2*i+1)
    ax2 = fig.add_subplot(nplots,2,2*i+2)
    ax2.set(yscale="log")

    qb  = q[:,i]
    qsb = q_s[:,i] 
    sns.distplot(qb , bins=50, color='b',kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.distplot(qsb, bins=50, color='r', kde=False, ax=ax1, norm_hist=True, label="s={0}".format(s_in[i]))

    # Compute and plot asymptotic distributions!
    var_mu_sb = 1/tf.abs(qAsb[i]) 
    var_mu_b  = 1/tf.abs(qAb[i]) 

    Eq_sb = -1 / var_mu_sb
    Eq_b  = 1 / var_mu_b

    Vq_sb = 4 / var_mu_sb
    Vq_b  = 4 / var_mu_b

    qsbx = np.linspace(np.min(qsb),np.max(qsb),1000)
    qsby = tf.math.exp(tfd.Normal(loc=Eq_sb, scale=tf.sqrt(Vq_sb)).log_prob(qsbx)) 
    sns.lineplot(qsbx,qsby,color='r',ax=ax1)
    qbx = np.linspace(np.min(qb),np.max(qb),1000)
    qby = tf.math.exp(tfd.Normal(loc=Eq_b, scale=tf.sqrt(Vq_b)).log_prob(qbx)) 
    sns.lineplot(qbx,qby,color='b',ax=ax1)


    #qx = np.linspace(np.min(q),np.max(q),1000)
    #qy = np.exp(tfd.Chi2(df=Npars).log_prob(qx))
    #sns.lineplot(qx,qy)
    sns.distplot(qb, color='b', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.distplot(qsb, color='r', kde=False, ax=ax2, norm_hist=True, label="s={0}".format(s_in[i]))
    sns.lineplot(qbx, qby,color='b',ax=ax2)
    sns.lineplot(qsbx,qsby,color='r',ax=ax2)

    #sns.lineplot(qx,qy)
    #ax.set_ylim(1./N,1.01*np.max(q)]))
ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
ax2.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)

plt.tight_layout()
fig.savefig("qsb_dists.png")

#plt.show()
