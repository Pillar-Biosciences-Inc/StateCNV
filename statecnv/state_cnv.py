import numpy as np
import scipy.stats as st  # type: ignore
from scipy.special import logsumexp  # type: ignore

from particles import distributions as dists
from particles import state_space_models as ssm

from particles import collectors as col
from particles.core import SMC
import time

def _extract_ratio_list(
    list_of_contiguous_segments_sample,
    list_of_contiguous_segments_normal,
    control_sample,
    control_normal,
):
    """ """
    n_segments = len(list_of_contiguous_segments_sample)
    ratio_list = []
    upweight = np.log(
        np.nanmedian((control_sample + 10) / (control_normal + 10))
    )

    for i in range(n_segments):
        myRatio = np.log(
            (list_of_contiguous_segments_sample[i] + 10)
            / (list_of_contiguous_segments_normal[i] + 10)
        )
        ratio_list.append(myRatio - upweight)
    return ratio_list


class StateSpaceGMM(ssm.StateSpaceModel):
    """
    This defines a state space model where the emissions distribution
    (observational likelihood) is a Gaussian distribution and the transition
    kernel is a Gaussian mixture model

    p(x_t|x_{t-1})=(1-p)*N(x_t,eps^2) + p*N(x_t,sigma^2)

    where eps << sigma. The idea is that you stay in the same category
    with high probability with a low probability of large jumps.
    """

    def PX0(self):
        return dists.Normal(loc=0.0, scale=1.0)

    def PX(self, t, xp):
        return dists.Mixture(
            [1 - self.p, self.p],
            dists.Normal(loc=xp, scale=self.scale1),
            dists.Normal(loc=xp, scale=self.scale2),
        )

    def PY(self, t, xp, x):
        return dists.Laplace(loc=x, scale=self.scale_emission / self.weights[t])


class GuidedSSM(StateSpaceGMM):

    def proposal0(self, data):
        return self.PX0()

    def proposal(self, t, xp, data):
        return dists.Mixture(
            [0.8, 0.2],
            dists.Normal(loc=xp, scale=self.scale1),
            dists.Normal(loc=data[t], scale=self.scale2),
        )


class StateCNV:
    def __init__(
        self,
        region_labels,
        n_particles=1000,
        p=0.02,
        scale_eps=0.0115,
        scale_sigma=0.12885,
        scale_emission=0.05,
        smallest_segment_length=5,
        qmc=False,
        resampling="stratified",
        ESSrmin=0.98,
    ):
        """
        This is a state space model that forms the basis of any state space
        callers for CNV. It relies on the state space model defined in
        StateSpaceGMM and uses code from the particles repository to perform
        inference using Sequential Monte Carlo. Depending on the caller,
        various methods will have to be called. However, all will share
        _sample_latent_space which generates samples from the posterior of
        the true CNV_ratio from the observed CNV ratio

        Parameters
        ----------
        n_particles : int,default=1000
            The number of particles used by the particle filter to
            charaacterize the state space posterior

        p : float,default=0.01
            The probability of a large jump (CNV change in true copies)

        scale_eps : float,default=.01
            The standard deviation of the tiny drift

        scale_sigma : flota,default=.3
            The standard deviation of the large jumps in latent space

        scale_emission : float,default=.1
            The standard deviation of the emission likelihood

        smallest_segment_length : int,default=5
            The smallest segment on which we will perform state space
            smoothing
        """
        self.n_particles = int(n_particles)
        self.p = float(p)
        self.scale_emission = float(scale_emission)
        self.scale_sigma = float(scale_sigma)
        self.scale_eps = float(scale_eps)
        self.smallest_segment_length = int(smallest_segment_length)

        self.region_labels = region_labels
        self.qmc = qmc
        self.resampling = resampling
        self.ESSrmin = ESSrmin

    def fit_call(self, cancer, normal, control_amplicon_indices, weights):
        n_regions = len(np.unique(self.region_labels))
        list_of_contiguous_segments_sample = []
        list_of_contiguous_segments_normal = []
        weights_list = []
        for i in range(n_regions):
            list_of_contiguous_segments_sample.append(
                cancer[self.region_labels == i]
            )
            list_of_contiguous_segments_normal.append(
                normal[self.region_labels == i]
            )
            weights_list.append(weights[self.region_labels == i])
        control_sample = cancer[control_amplicon_indices]
        control_normal = normal[control_amplicon_indices]

        self._sample_latent_space(
            list_of_contiguous_segments_sample,
            list_of_contiguous_segments_normal,
            control_sample,
            control_normal,
            weights_list,
        )
        return np.mean(self.samples, axis=1)

    def _sample_latent_space(
        self,
        list_of_contiguous_segments_sample,
        list_of_contiguous_segments_normal,
        control_sample,
        control_normal,
        weights_list,
    ):
        """
        This generates the samples from the latent space trajectory, that
        is given a state space model

        p(x_0)
        p(x_t|x_{t-1})
        p(y_t|x_t)

        This generates samples of p(x|y_{0:T}).

        In this case, x are the true copy number ratios and y are the
        observed copy number ratios
        """
        n_segments = len(list_of_contiguous_segments_sample)
        n_particles = self.n_particles

        ratio_list = _extract_ratio_list(
            list_of_contiguous_segments_sample,
            list_of_contiguous_segments_normal,
            control_sample,
            control_normal,
        )

        samples_list = []
        index_segments = []
        logLts = []

        startTime = time.time()
        essVals = []
        logLtList = []

        for i in range(n_segments):
            ratios = ratio_list[i]
            n_amplicons = len(ratios)
            index_segments.append(i * np.ones(n_amplicons))
            weights = weights_list[i]
            if n_amplicons >= self.smallest_segment_length:
                model = GuidedSSM(
                    p=self.p,
                    scale1=self.scale_eps,
                    scale2=self.scale_sigma,
                    scale_emission=self.scale_emission,
                    weights=weights,
                )
                X = [ratios[i] for i in range(len(ratios))]

                fk_model1 = ssm.GuidedPF(ssm=model, data=X)

                # Perform sequential monte carlo
                pf = SMC(
                    fk=fk_model1,
                    N=self.n_particles,
                    qmc=self.qmc,
                    resampling=self.resampling,
                    ESSrmin=self.ESSrmin,
                    collect=[col.Moments(), col.LogLts(), col.ESSs()],
                    store_history=True,
                )
                pf.run()
                logLts.append(pf.summaries.logLts[-1])
                essVals.append(np.array(pf.summaries.ESSs))

                # Perform backward smoothing to compute trajectories
                smoothed_posterior = pf.hist.backward_sampling_ON2(
                    self.n_particles
                )
                samples = np.zeros((len(smoothed_posterior), self.n_particles))
                for i in range(n_amplicons):
                    samples[i] = smoothed_posterior[i]
                samples_list.append(samples)
                logLtList.append(np.array(pf.summaries.logLts))
            else:
                out = np.zeros((n_amplicons, n_particles))
                for j in range(n_amplicons):
                    out[j, :] = ratios[j]
                samples_list.append(out)

        self.elapsedTime = time.time() - startTime
        self.essVals = np.concatenate(essVals)
        self.samples = np.vstack(samples_list)
        self.index_segments = np.concatenate(index_segments)
        self.logLts = np.array(logLts)
        self.logLtList = logLtList

    def _diagnose(
        self,
        list_diagnoses,
        return_full=False,
        scale_weights=None,
        prior_probs=None,
    ):
        """
        This generates a diagnosis given a list of diagnoses (that is a
        list of expected behavior if an individual has disease
        list_diagnoses[i]). Note that we DO NOT APPEND A DIAGNOSIS FOR "
        NO PROBLEM", we expect you to do that yourself.

        Parameters
        ----------
        list_diagnoses : list[np.array]
            A list of expected behavior for each potential diagnosis

        return_full : bool,default=False
            Whether to return the full posterior and likelihood or just the
            estimated ratio + call

        scale_weights : np.array(self.samples.shape[0]),default=None
            This allows us to upweight/downweight specific amplicons as
            being predictive of specific diseases

        prior_probs : np.array(len(list_diagnoses)),default=None
            The prior probabilities of the different diseases

        Returns
        -------
        X_hat :  np.array-like,(n_amplicons,)
            The copy number ratio estimate

        call : int
            The index of the called diagnosis

        posterior : np.array-like,(n_diagnoses,)
            Optional: the posterior probabilities of each diagnosis

        log_evidence : float
            This is the evidence for a specific disease (note this is not
            the evidence for the quality of the sample, rather the quality
            of the diagnosis)
        """
        log_evidence = 0
        n_diagnoses = len(list_diagnoses)
        n_amplicons = len(list_diagnoses[0])
        posterior = np.zeros(n_diagnoses)

        if scale_weights is None:
            scale_weights = np.ones(n_amplicons)
        scales = scale_weights * self.scale_emission

        if prior_probs is None:
            prior_probs = 1 / n_diagnoses * np.ones(n_diagnoses)

        for i in range(self.n_particles):
            trajectory = self.samples[:, i]
            log_likelihoods = np.zeros((n_amplicons, n_diagnoses))
            for j in range(n_diagnoses):
                log_likelihoods[:, j] = st.norm.logpdf(
                    trajectory, loc=list_diagnoses[j], scale=scales
                )

            log_likelihoods[log_likelihoods < -80] = -80

            log_likelihood = np.sum(log_likelihoods, axis=0)
            log_posterior_unnormalized = log_likelihood + np.log(prior_probs)
            log_evidence_temp = logsumexp(log_posterior_unnormalized)
            log_posterior = log_posterior_unnormalized - log_evidence_temp
            posterior_temp = np.exp(log_posterior)
            posterior += posterior_temp / self.n_particles
            log_evidence += log_evidence_temp / self.n_particles

            log_likelihoods += log_likelihood / self.n_particles

        call = np.argmax(posterior)
        X_hat = np.mean(self.samples, axis=1)

        if return_full:
            return X_hat, call, posterior, log_likelihoods, log_evidence
        else:
            return X_hat, call

    def _hierarchical_diagnose(
        self,
        list_diagnoses,
        groupings,
        return_full=False,
        scale_weights=None,
        prior_probs=None,
    ):
        groups = np.unique(groupings)
        n_groups = len(np.unique(groupings))
        n_diagnoses = len(prior_probs)
        p_d_g = np.zeros(n_diagnoses)
        for i in range(n_diagnoses):
            group = groupings[i]
            p_d_g[i] = prior_probs[i] / np.sum(prior_probs[groupings == group])

        X_hat, call, posterior, log_likelihoods, log_evidence = self._diagnose(
            list_diagnoses, scale_weights=scale_weights, return_full=True
        )

        group_probs = np.zeros(n_groups)
        for g in range(n_groups):
            selected_probs = log_likelihoods[:, groupings == groups[g]]
            selected_priors = p_d_g[groupings == groups[g]]
            selected_log_posteriors = (
                np.log(selected_priors + 1e-8) + selected_probs
            )
            group_probs[g] = logsumexp(selected_log_posteriors)

        group_probs = group_probs / np.sum(group_probs)
        call = np.where(np.amax(group_probs) == group_probs)[0]

        return call, group_probs

    def _detect_jumps(self, delta, alpha=0.05, k=5):  # idx_breaks = []
        ss_sub = np.exp(self.samples[::k])  # np.exp(self.samples[idxs_breaks])
        # difference_tp1_m_t = ss_sub[1:] - ss_sub[:-1]
        # abs_difference_tp1_m_t = np.abs(ss_sub[1:] - ss_sub[:-1])
        # detect_jump = abs_difference_tp1_m_t > delta
        idx_diff = 1.0 * (np.abs(ss_sub[1:] - ss_sub[:-1]) > delta)
        p_diff = np.mean(idx_diff, axis=1)
        idxs_reduced = np.where(1.0 * (p_diff > (1 - alpha)))[0]
        idxs = idxs_reduced * k
        if len(idxs) == 1:
            return []
        return idxs

    def cnv_by_region(self, delta, alpha=0.05, k=5):
        idx_list_new = self._detect_jumps(delta, alpha, k)

        cnv_list = {}
        ss = np.exp(self.samples)
        x = np.arange(ss.shape[0])

        if len(idx_list_new) > 0:
            n_changes = len(idx_list_new)
            idx_list_tmp = idx_list_new.copy()
            idx_list_tmp = np.insert(idx_list_tmp, 0, 0)
            idx_list_tmp = np.append(idx_list_tmp, len(self.region_labels))

            for j in range(n_changes + 1):
                start = idx_list_tmp[j]
                end = idx_list_tmp[j + 1]
                meds = np.mean(ss, axis=1)
                idxs = (x > start - 0.1) * (x < end + 0.1)
                rr = np.median(meds[idxs])
                segment = {"ratio": rr, "start": start, "end": end}
                cnv_list[j] = segment
        else:
            segment = {"ratio": 1, "start": 0, "end": self.samples.shape[0]}
            cnv_list[0] = segment
        return cnv_list

    def _evaluate_probability(self, condition_function):
        """
        Given a condition measured by the function condition_function, this
        method evaluates the posterior probability of this

        Parameters
        ----------
        condition_function : function,f:(n_amplicons)-> {0,1}
            Given an array of values (estimated copy ratio) returns 1 if
            true and 0 if false

        Returns
        -------
        prob : float
            The posterior probability of a specific condition
        """
        prob = 0
        for i in range(self.n_particles):
            trajectory = self.samples[:, i]
            prob += condition_function(trajectory) / self.n_particles
        return prob
