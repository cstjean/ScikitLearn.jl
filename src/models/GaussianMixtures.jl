importall ScikitLearnBase
using GaussianMixtures
import GaussianMixtures: GMM, logsumexp, llpg, kind

n_components(gmm::GMM) = size(gmm.μ, 1)

# We set d=1, but that's OK, we'll change it in fit!
GMM(; n_components=1, kind=:diag) = GMM(n_components, 2; kind=kind)
is_classifier(::GMM) = false
clone(gmm::GMM) =
    GMM(; n_components=ScikitLearn.n_components(gmm),
        kind=GaussianMixtures.kind(gmm))

function get_params(gmm::GMM)
    return Dict(:n_components=>n_components(gmm))
end

function set_params!(gmm::GMM; params...)
    for (param, val) in params
        if param==:n_components
            # We don't have to update Σ, it'll be wiped in fit! anyway
            gmm.μ = zeros(val, 1)
        else
            throw(ArgumentError("Bad parameter: $param"))
        end
    end
    gmm
end

function Base.copy!(gmm_dest::GMM, gmm_src::GMM)
    # shallow copy - used below
    for f in fieldnames(gmm_dest)
        setfield!(gmm_dest, f, getfield(gmm_src, f))
    end
end

function fit!(gmm::GMM, X::AbstractMatrix, y=nothing)
    n = n_components(gmm)
    # Creating a temporary is not great, but it's negligible in the grand
    # scheme of thing.  We'd just need a slight refactor in
    # GaussianMixtures/src/train.jl to avoid it
    gmm_temp = GMM(n, X; kind=kind(gmm))
    copy!(gmm, gmm_temp)
    gmm
end

predict_log_proba(gmm::GMM, X) = log(gmmposterior(gmm, X)[1])
predict_proba(gmm::GMM, X) = gmmposterior(gmm, X)[1]
predict(gmm::GMM, X) =
    # This is just `argmax(axis=2)`. It's very verbose in Julia.
    ind2sub(size(X), vec(findmax(predict_proba(gmm, X), 2)[2]))[2]

""" `density(gmm::GMM, X)` returns `log(P(X|\mu, \Sigma))` """
density(gmm::GMM, X) =
    squeeze(logsumexp(broadcast(+, llpg(gmm, X),
                                reshape(log(gmm.w), 1, length(gmm.w))), 2), 2)

# score_samples is underspecified by the scikit-learn API, so we're more or
# less free to return what we want
# http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html#sklearn.mixture.GMM
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
"""    score_samples(gmm::GMM, X::Matrix)
Return the per-sample likelihood of the data under the model. """
score_samples(gmm::GMM, X) = density(gmm, X)

# Used for cross-validation. Higher is better.
score(gmm::GMM, X) = GaussianMixtures.avll(gmm, X)
