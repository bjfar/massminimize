"""Monte Carlo simulations of GAMBIT ColliderBit signal region selection"""

import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import massminimize as mm
import matplotlib.pyplot as plt
import seaborn as sns

#tf.random.set_seed(np.random.randint(0,1000))

def eCDF(x):
    """Get empirical CDF of some samples"""
    return np.arange(1, len(x)+1)/float(len(x))

Nsig = 3 # Number of signal regions
Npars = 1
N = int(1e4)
Amp = 100
#TS = 'qraw'
TS = 'qb'
#TS = 'qmax'
#TS = 'qbmax'

# We need a more complicated model this time so that the SR selection criteria makes sense.
# I think searching for bump of fixed width, with SR search bins of fixed width, will be
# the simplest case.

# True signal parameters
A = tf.constant(Amp,dtype=float) # signal amplitude (total expected number of events)

# SR definitions
LH = 0
RH = 100
W = 10

# Signal peaks in parameter space
sigc = tf.constant(np.linspace(LH,RH,Nsig,endpoint=False) + (RH-LH)/Nsig/2.,dtype=float)

#xT = tf.constant(sigc[0],dtype=float)
xT = tf.constant(1,dtype=float)
print("xT:",xT)

tag = "{0}_randb_A{1}_xT{2}".format(TS,Amp,int(xT.numpy()))
suffix = "_Nbins{0}_Ntrials{1}_{2}_scaledsigs".format(Nsig,N,tag)

# Signal predictions for all SRs as a function of scaling parameter
def signal(mu):
    #sig = A*tf.exp(-tf.abs(mu-c)/W)
    # Actually, those super sharp peaks make it hard for the optimiser to find the peak (it keeps jumping over it because it is non-differentiable)
    # So smooth out the peaks a bit.
    absdiff = tf.sqrt(tf.abs(mu-sigc)**2 + 10)
    sig = A*tf.exp(-absdiff/W)
    #mask = np.ones(mu.shape)
    #try:
    #    mask[mu.numpy()<10] = 0 # always predict zero signal for part of parameter space near mu=0
    #except AttributeError:
    #    mask[mu<10] = 0
    #No good, too hard to fit
    # But, do need to deal with boundaries of parameter space,
    # otherwise best fit can be at infinity when zero signal is the best fit
    sLH = A*tf.exp(-tf.sqrt(tf.abs(LH-sigc)**2 + 10)/W)
    sRH = A*tf.exp(-tf.sqrt(tf.abs(RH-sigc)**2 + 10)/W)
    Lmask = np.ones(mu.shape,dtype=np.bool)
    try:
        Lmask[mu.numpy()<LH] = 0
    except AttributeError:
        Lmask[mu<LH] = 0
    Rmask = np.ones(mu.shape,dtype=np.bool)
    try:
        Rmask[mu.numpy()>RH] = 0
    except AttributeError:
        Rmask[mu>RH] = 0
   
    outsig = sig * tf.constant(Lmask*Rmask,dtype=float) \
         + sLH * tf.constant(~Lmask,dtype=float) \
         + sRH * tf.constant(~Rmask,dtype=float)
    return outsig
    #return sig

# randomised backgrounds
b = tf.constant(np.random.randint(10,4000,Nsig),dtype=float)
print("b:",b)
s = signal(xT)
print("s:",s)

# Later we will want separate parameters for each signal region
# Can broadcast this like so:
x_many = tf.constant([xT.numpy() for i in range(Nsig)],dtype=float)
s_many = signal(x_many)
print("s_many:", s_many) # should match s since all 'x' parameters are the same

# Bunch of independent Poisson-->(now Normal!) distributions
poises0  = tfd.Normal(loc=b, scale=tf.sqrt(b))
poises0s = tfd.Normal(loc=s+b, scale=tf.sqrt(b)) # keep variance independent of s for simplicity

# Generate background-only pseudodata to be fitted
samples0 = poises0.sample(N)

# Generate signal+background pseudodata to be fitted
samples0s = poises0s.sample(N)

print("samples0:",samples0)
print("samples0s:",samples0s)

# Let's plot some of these so we feel confident that the generated data makes sense
# This is a bit tricky since the data isn't exactly "bins" in our parameter anymore
# However the signal sensitivities are concentrated around certain parameter values,
# so there is some intuitive notion by which we can assign the counts to those "places"

# Ok screw it just pick the bin nearest the predict centre of the signal for now.
# Result should be the same, can add explicit calculation later.
def selector(pos):
   #centres = sigc 
   #dist = tf.abs(centres - pos)
   #return tf.cast(tf.argmin(dist,axis=1),tf.int32)
   # NEW VERSION
   # based on L_{s+b} / L_b sensitivity
   s = signal(pos)
   poises_sb = tfd.Normal(loc=s+b, scale=tf.sqrt(b))
   qsb_expb = -2*poises_sb.log_prob(b)
   qb_expb  = -2*poises0.log_prob(b)
   sens = qsb_expb - qb_expb
   # select SR with largest "sens"
   return tf.cast(tf.argmax(sens,axis=1),tf.int32)

pos = tf.expand_dims(tf.constant(np.linspace(LH,RH,1000),dtype=float),1)
sel = selector(pos)
print("selector:", sel)

# Use "sel" to work out contiguous regions with common SR selection in the parameter space
lh=[]
rh=[]
delta=1e-6
for i in range(Nsig):
    m=sel.numpy()==i
    print("m:",m)
    if tf.reduce_any(m):
       lh += [pos[m][0,0]]
       rh += [pos[m][-1,0]]
    elif i==0:
       lh += [0]
       rh += [delta]
    else:
       lh += [rh[-1]]
       rh += [rh[-1]+delta]
lh = np.array(lh)
rh = np.array(rh)
w = rh - lh
print("lh:",lh)
print("rh:",rh)
print("w:",w)

# For now these also act as the boundaries of the signal region selection
# TODO: Need to replace this with proper "most sensitive" selection
# DONE ABOVE
#w = (RH-LH)/Nsig
#lh = np.array([(w*i+LH) for i in range(Nsig)])
#rh = np.array([(w*(i+1)+LH) for i in range(Nsig)])

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
nsig = samples0s[0] - b
# Plot signal predictions
mu = np.linspace(LH,RH,1000)
sigs = signal(mu[:,np.newaxis])
print("sigs:",sigs)
ax.plot(mu,sigs)
# Plot zero line
ax.axhline(0,c='k')
# Add error bars to show the background variance
ax.errorbar((lh+rh)/2., 0*nsig, yerr=np.sqrt(b), fmt='none', c='b', elinewidth=rh[0]-lh[0], alpha=0.3)
ax.plot(list(lh)+[rh[-1]],list(nsig)+[nsig[-1]],drawstyle='steps-post',c='b')
plt.tight_layout()
fig.savefig("bump_data{0}.png".format(suffix))

# Version scaled against variance
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
# Plot zero line
ax.axhline(0,c='k')
# Add error bars to show the background variance
#ax.errorbar((lh+rh)/2., 0*nsig, yerr=np.sqrt(b), fmt='none', c='b', elinewidth=rh[0]-lh[0], alpha=0.3)
nscaled = nsig/np.sqrt(b)
ax.plot(list(lh)+[rh[-1]],list(nscaled)+[nscaled[-1]],drawstyle='steps-post',c='b')
plt.tight_layout()
fig.savefig("bump_data_scaled{0}.png".format(suffix))

# Best expected signal region likelihood

def LogLike(p,data=samples0s):
    s = signal(p)
    #print("p:",p)
    #print("s:",s)
    poises_sb = tfd.Normal(loc=s+b, scale=tf.sqrt(b))
    L = -2*poises_sb.log_prob(samples0s)
    return L

# Likelihoods for all signal regions
def Lall(p1,data=samples0s,allTS=False):
    s = signal(p1)
    #print("p1:",p1)
    #print("s:",s)
    poises_sb = tfd.Normal(loc=s+b, scale=tf.sqrt(b))
    #print("b:", b)
    #print("rate:", p1+b)
    #print("poises_sb:", poises_sb)
    qsb = -2*poises_sb.log_prob(data)
    #print("qsb:", qsb)
    qb  = -2*poises0.log_prob(data)
    qmin = -2*poises_sb.log_prob(s+b)
    qbmin  = -2*poises0.log_prob(b)
    #print("qmin:", qmin)

    LRqraw = qsb
    LRqb   = qsb - qb
    LRqmax  = qsb - qmin
    LRqbmax = qsb - qmin - (qb - qbmin)

    if TS=='qraw':
        LLRs = LRqraw
    elif TS=='qb':
        LLRs = LRqb
    elif TS=='qmax':
        LLRs = LRqmax
    elif TS=='qbmax':
        LLRs = LRqbmax      
    else:
        raise ValueError("Unknown TS selected!")

    # Turns out the fit is unstable if we do the selection here (probably because the problem
    # becomes non-convex). So instead find best fit point for *all* SRs individually, and
    # then select one for each trial afterwards.
 
    if allTS:
        return LRqraw, LRqb, LRqmax, LRqbmax
    else:
        return LLRs

# Parameter to fit. One for each signal region
#p_in = tf.Variable(x*np.ones((N,Nsig), dtype='float32'), dtype=float, name='p1')

# Bounded version: we cannot directly implement bounds, so we need to do it implicitly via a
# change of variables like:
# c = a + (b - a) / (1 + exp(-d))
# Then scan d, imposes bounds [a,b] on c
# So need different starting guesses 

p_in = tf.Variable(s*np.zeros((N,1), dtype='float32')+2., dtype=float, name='p1')

# Let's also do the combined fit at the same time, for comparison
# The full likelihood surface is nasty so we need to start from the basin of the correct minimum
# Do a quick grid scan to find a good starting guess
p_guess_in = np.linspace(LH+delta,RH-delta,Nsig*10)
p_guess = tf.constant(p_guess_in[:,np.newaxis,np.newaxis], dtype=float, name='p_guess')
L_grid_all = LogLike(p_guess)
print("L_grid_all:", L_grid_all)
L_grid_max_i = tf.cast(tf.argmin(tf.reduce_sum(L_grid_all,axis=2),axis=0),tf.int32)
print("L_grid_max_i:", L_grid_max_i)
# Select best starting guesses from grid
p_best_guess = tf.constant(p_guess_in[L_grid_max_i], dtype=float)
print("p_best_guess:",p_best_guess)

# Inverse of scaling operation to get sensible starting guess for combined fit (near true value)
pc_start = -tf.math.log( (RH-LH)/(p_best_guess-LH) - 1 )
prec = LH + (RH-LH)/(1 + tf.exp(-pc_start)) # check calculation
print("recovered start guess:", prec) 
p_comb = tf.Variable(tf.expand_dims(pc_start,axis=1), dtype=float, name='pcomb') # Scaled version
#p_comb = tf.Variable(tf.expand_dims(p_best_guess,axis=1), dtype=float, name='pcomb') # Unscaled version

print("p_in:",p_in)
print("p_comb:",p_comb)

# Do SR selection fits
def lossSR(pars):
    # Bounded version:
    p1 = lh + (rh-lh)/(1 + tf.exp(-pars[0]))
    #print("p_in:", p_in)
    #print("p1:", p1)
    
    LLR = Lall(p1)
    total_loss = tf.math.reduce_sum(LLR)
    return total_loss, [p1], None

[p1], none = mm.optimize([p_in],lossSR,0.001,tol=1e-8,grad_tol=1e-4,conv_from_trans_pars=True)

# Do combined fits
def lossComb(pars):
    # Full combined likelihood
    pc = LH + (RH-LH)/(1 + tf.exp(-pars[0]))
    #pc = pars[0]

    #print("pc:", pc)
    Lcomb = LogLike(pc)
    
    #print("LLR:", LLR)
    #print("Lcomb:", Lcomb)
    total_loss = tf.math.reduce_sum(Lcomb)
    return total_loss, [pc], None

[pc], none = mm.optimize([p_comb],lossComb,0.001,tol=1e-8,grad_tol=1e-4,conv_from_trans_pars=True)

# First plot distribution of LLRs for fits to individual signal regions
# (TODO: this currently combines them all into one plot. Should still
# be chi2 since they individually should be chi2. But probably good to
# do separate plots, less confusing)

LLR_bf = Lall(p1)
print("LLR_bf.shape:", LLR_bf.shape)

Lcomb_bf_sep = LogLike(tf.expand_dims(pc,axis=0))
print("Lcomb_bf_sep.shape:", Lcomb_bf_sep.shape)
# This is still all likelihood components separately. Need to add them up.
Lcomb_bf = tf.transpose(tf.reduce_sum(Lcomb_bf_sep,axis=2))
print("Lcomb_bf.shape:", Lcomb_bf.shape)

all_truepar = tf.broadcast_to(xT,shape=(N,Nsig))
print("all_truepar.shape:", all_truepar.shape)
MLLR = (Lall(all_truepar) - LLR_bf)
#MLLR = Lall(all_truepar) # Try without subtracting best fit
print("MLLR.shape:", MLLR.shape)

# Combined MLLR
LcombT_sep = LogLike(tf.expand_dims(xT,axis=0))
print("LcombT_sep:",LcombT_sep)
LcombT = tf.transpose(tf.reduce_sum(LcombT_sep,axis=1))
MLLR_comb = LcombT - Lcomb_bf[:,0]
print("MLLR_comb.shape:", MLLR_comb.shape)

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
ax1.set(yscale="log")
sns.distplot(MLLR[0], bins=50, kde=False, ax=ax1, norm_hist=True, label="MLLR")
q = np.linspace(np.min(MLLR[0]),np.max(MLLR[0]),1000)
chi2 = tf.math.exp(tfd.Chi2(df=1).log_prob(q)) 
sns.lineplot(q,chi2,color='b',ax=ax1)

plt.tight_layout()
fig.savefig("ind_fits{0}.png".format(suffix))

# Now we need to select the signal region depending on the parameter values
# First, select based on the *test* point. This will always choose the
# same signal region, no matter the data, so it should work just fine.
truepar = tf.broadcast_to(xT,shape=(N,1))
selected = tf.expand_dims(selector(truepar),1)
print("selected:", selected)
ordinals = tf.reshape(tf.range(MLLR.shape[0]), (-1,1))
idx = tf.stack([ordinals, selected], axis=-1)
MLLRsel = tf.gather_nd(MLLR, idx)

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
ax1.set(yscale="log")
sns.distplot(MLLR_comb, bins=50, kde=False, ax=ax1, norm_hist=True, label="Combined")
sns.distplot(MLLRsel, bins=50, kde=False, ax=ax1, norm_hist=True, label="Selected")
q = np.linspace(np.min(MLLRsel),np.max(MLLRsel),1000)
chi2 = tf.math.exp(tfd.Chi2(df=1).log_prob(q)) 
sns.lineplot(q,chi2,color='b',ax=ax1)

plt.tight_layout()
fig.savefig("MLLR_sel{0}.png".format(suffix))

# Plot cut-paste likelihood surface for a couple of trials
ptest = tf.expand_dims(tf.constant(np.linspace(LH,RH-0.00001,1000),dtype=float),1)
#print("ptest:",ptest)
print("samples0s[0]:", samples0s[0])
Lrawtest, Lbtest, Lmaxtest, Lbmaxtest = Lall(ptest,data=samples0s[0],allTS=True)

selected = tf.expand_dims(selector(ptest),1)
print("selected.shape:", selected.shape)
ordinals = tf.reshape(tf.range(ptest.shape[0]), (-1,1))
idx_cut = tf.stack([ordinals, selected], axis=-1)
print("idx_cut:",idx_cut)

Lcuttests =[]
for Ltest in [Lrawtest, Lbtest, Lmaxtest, Lbmaxtest]:
   Lcuttests += [tf.gather_nd(Ltest, idx_cut)]
#print("Lcuttest:", Lcuttest)

# Get BF points (check value on plotted surface)
print("LLR_bf:",LLR_bf)
iatbf = tf.expand_dims(tf.cast(tf.math.argmin(LLR_bf,axis=1),tf.int32),1) # Pick global maximum "likelihood" from across restricted SR fits
print("iatbf.shape:", iatbf.shape)
ordinals = tf.reshape(tf.range(LLR_bf.shape[0]), (-1,1))
idx_bf = tf.stack([ordinals, iatbf], axis=-1)
print("idx_bf.shape:",idx_bf.shape)
BFLLR = tf.gather_nd(LLR_bf, idx_bf)
BFpar = tf.gather_nd(p1, idx_bf)
print("BFLLR.shape:", BFLLR.shape)

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
# Complete likelihood for one signal region
#sns.lineplot(ptest[:,0],Ltest[:,0]-BFLLR[0,0],color='r',ax=ax1,'SR[0]')
# Full joint likelihood (combined independent signal regions)
Lcomb = tf.reduce_sum(Lrawtest,axis=1)
sns.lineplot(ptest[:,0],Lcomb-tf.reduce_min(Lcomb),color='g',ax=ax1,label='Comb.')
# Cut-paste likelihoods
for Lcut, c, lab in zip(Lcuttests,['b','c','m','r'],['qraw','qb','qmax','qbmax']):
   sns.lineplot(ptest[:,0],Lcut[:,0]-tf.reduce_min(Lcut[:,0]),color=c,ax=ax1,label=lab)
# Offset?
offset = signal(xT)[0]**2 / b[0]
print("offset:",offset)
ax1.axhline(offset.numpy(),color='k')

# Best fit points
ax1.scatter(BFpar[0,0].numpy(),0,c='k',s=50,marker='o',label='BF') # Global best fit
ax1.scatter(p1[0,:],LLR_bf[0,:]-BFLLR[0,0],color='r',marker='o',s=20) # SR restricted best fits
ax1.scatter(pc[0,:],0,color='g',marker='o',s=15) # Combined best fit
ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
plt.tight_layout()
fig.savefig("cutLRsurface{0}.png".format(suffix))

# Scaled version of plot to focus on "good-fitting" regions
fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
sns.lineplot(ptest[:,0],Lcomb-tf.reduce_min(Lcomb),color='g',ax=ax1,label='Comb.')
for Lcut, c, lab in zip(Lcuttests,['b','c','m','r'],['qraw','qb','qmax','qbmax']):
   sns.lineplot(ptest[:,0],Lcut[:,0]-tf.reduce_min(Lcut[:,0]),color=c,ax=ax1,label=lab)
ax1.axhline(offset.numpy(),color='k')
ax1.scatter(BFpar[0,0].numpy(),0,c='k',s=50,marker='o',label='BF') # Global best fit
ax1.scatter(p1[0,:],LLR_bf[0,:]-BFLLR[0,0],color='r',marker='o',s=20) # SR restricted best fits
ax1.scatter(pc[0,:],0,color='g',marker='o',s=15) # Combined best fit
ax1.legend(loc=1, frameon=False, framealpha=0, prop={'size':10}, ncol=1)
plt.tight_layout()
ax1.set_ylim(0,25)
fig.savefig("cutLRsurface_yto5sigma{0}.png".format(suffix))


# Now select the signal regions separately for test point and BF point
# TODO: Actually damn, we don't have the data we need for this, because
# we don't have much idea what the "combined" likelihood surface looks
# like, so we don't know that the "combined" BF point is one of the BF points
# for the individual SRs (and probably isn't; the cut-and-paste likelihood
# surface will be very bizarre).
# Gah, not sure if that is solvable. Need a proper global minimiser to find
# these "combined" BF points? No way to get them from a GD minimiser like TF?
# Would have to minimise in each SR "selection region" individually, and then
# see which of those is the global minima? I guess that can work. Bit tricky
# to set up though.
# ...or is it? Just need to restrict the "individual SR" fits to those regions
# where they are selected? Should be fine if they are contiguous regions each
# with a convex likelihood surface.

qattrue = tf.gather_nd(Lall(all_truepar), idx)[:,0] # Select SR for test point (same for all data)
print("qattrue:", qattrue)
qatbf   = tf.reduce_min(LLR_bf,axis=1) # Pick global maximum "likelihood" from across restricted SR fits
print("qatbf:", qatbf)

MLLRsepsel = qattrue - qatbf
print("MLLRsepsel:", MLLRsepsel)

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
ax1.set(yscale="log")
sns.distplot(MLLR_comb, bins=50, kde=False, ax=ax1, norm_hist=True, label="Comb.")
sns.distplot(MLLRsepsel, bins=50, kde=False, ax=ax1, norm_hist=True, label="MLLR")
q = np.linspace(np.min(MLLRsepsel),np.max(MLLRsepsel),1000)
chi2 = tf.math.exp(tfd.Chi2(df=1).log_prob(q)) 
sns.lineplot(q,chi2,color='b',ax=ax1)
#chi2n = tf.math.exp(tfd.Chi2(df=20).log_prob(q)) 
#sns.lineplot(q,chi2n,color='r',ax=ax1)

plt.tight_layout()
fig.savefig("MLLR_sep_sel{0}.png".format(suffix))

# CDF version
fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
ax1.set(yscale="log")
MLLRsort_raw = np.sort(MLLRsepsel.numpy())
print("MLLRsort_raw:", MLLRsort_raw)
# Numerical errors can lead to MLLRsort<0
# Need to fix these.
MLLRsort = MLLRsort_raw * (MLLRsort_raw>=0)
print("MLLRsort:", MLLRsort)
MLLR_comb_sort_raw = np.sort(MLLR_comb.numpy())
MLLR_comb_sort = MLLR_comb_sort_raw * (MLLR_comb_sort_raw>=0)

pvals = 1-eCDF(MLLRsort)
print("pvals:", pvals)
pvals_comb = 1-eCDF(MLLR_comb_sort)
sns.lineplot(MLLR_comb_sort,pvals_comb, color='g', label="Comb.")
sns.lineplot(MLLRsort,pvals, color='k', label="Select.")
q = np.linspace(np.min(MLLRsepsel),np.max(MLLRsepsel),1000)
chi2cdf = tfd.Chi2(df=1).cdf(q)
sns.lineplot(q,1-chi2cdf,color='b',ax=ax1)
#chi2cdfmin = tfd.Chi2(df=1).cdf(q)**Nsig # CDF of max of Nsig chi^2 random variables 
#sns.lineplot(q,chi2cdfmin,color='r',ax=ax1)

# Compute mapping from chi-sqaure based significance estimate to
# true significance

plt.tight_layout()
fig.savefig("MLLR_sep_sel_cdf{0}.png".format(suffix))

# Significance correction version
# CDF version
fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)

ax1.grid()

ax1.set_xticks([-1,0,1,2,3,4,5], minor=False)
ax1.set_yticks([0,1,2,3,4,5], minor=False)
ax1.xaxis.grid(True, which='major')
ax1.xaxis.grid(False, which='minor')
ax1.yaxis.grid(True, which='major')
ax1.yaxis.grid(False, which='minor')

x = np.linspace(0,5,10)
sns.lineplot(x,x,color='b',ax=ax1)

for mllr,c,alph,lab in zip([MLLR_comb_sort,MLLRsort],['g','k'],[0.2,0.5],['Comb.','Select.']):
   apvals = 1 - tfd.Chi2(df=1).cdf(mllr)
   pvals = 1-eCDF(mllr)
   in_sig_raw = -tfd.Normal(loc=0,scale=1).quantile(apvals)
   out_sig_raw = -tfd.Normal(loc=0,scale=1).quantile(pvals)
   in_sig = np.array(in_sig_raw.numpy(),dtype=np.float64)
   out_sig = np.array(out_sig_raw.numpy(),dtype=np.float64)
   in_sig[np.isneginf(in_sig)] = -1e10
   out_sig[np.isposinf(out_sig)] = 1e10
   sns.lineplot(in_sig,out_sig, color=c, label=lab)
  
   inx = [1,2,3]
   print("in_sig:",in_sig)
   print("out_sig:", out_sig)
   sigf = sp.interpolate.interp1d([-1e99]+list(in_sig)+[1e99],[-1e99]+list(out_sig)+[1e99],copy=False)
   correctsig = sigf(inx)
   print("correctsig:", correctsig)
   
   #ax1.vlines(inx,ymin=[0,0,0],ymax=correctsig,alpha=alph,color=c)
   ax1.hlines(correctsig,xmin=inx,xmax=4,alpha=alph,color=c)
 
ax1.set_xlim(0,4)
ax1.set_ylim(0,4)

ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(correctsig)

ax1.set_xlabel("Chi2-based significance")
ax2.set_ylabel("Simulated signifiance")

plt.tight_layout()
fig.savefig("MLLR_sep_sel_sig_cor{0}.png".format(suffix))


