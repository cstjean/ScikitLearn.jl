importall ScikitLearnBase
using GaussianMixtures
import GaussianMixtures: GMM, logsumexp, llpg, kind

# We set d=1, but that's OK, we'll change it in fit!
GMM(; n=1, kind=:diag) = GMM(n, 2; kind=kind)
is_classifier(::GMM) = true

function reinitialize!(gmm::GMM, n, d)
    # Taken from the GMM(n, d) constructor
    if kind(gmm) == :diag
        gmm.Σ = ones(n, d)
    elseif kind(gmm) == :full
        gmm.Σ = UpperTriangular{Float64, Matrix{Float64}}[UpperTriangular(eye(d)) for i=1:n]
    else
        error("Unknown kind")
    end
    rng = MersenneTwister(42)
    #gmm.μ = rand(rng, n, d)
    gmm.hist = [History(@sprintf "Initialization n=%d, d=%d, kind=%s" n d kind)]
end

function Base.copy!(gmm_dest::GMM, gmm_src::GMM)
    # shallow copy
    for f in fieldnames(gmm_dest)
        setfield!(gmm_dest, f, getfield(gmm_src, f))
    end
end

function fit_bad!(gmm::GMM, X)
    # Now that we know d, reinitialize the matrices
    reinitialize!(gmm, size(gmm.μ, 1), size(X, 2))
    em!(gmm, X)
    gmm
end 

function fit!(gmm::GMM, X::AbstractMatrix)
    n = size(gmm.μ, 1)
    # Creating a temporary is not great, but it's negligible in the grand
    # scheme of thing.  We'd just need a slight refactor in
    # GaussianMixtures/src/train.jl to avoid it
    gmm_temp = GMM(n, X; kind=kind(gmm))
    copy!(gmm, gmm_temp)
    gmm
end

predict_log_proba(gmm::GMM, X) = llpg(gmm::GMM, x::Matrix)
predict_proba(gmm::GMM, X) = exp(predict_log_proba(gmm, X))

density(gmm::GMM, X) = squeeze(logsumexp(broadcast(+, llpg(gmm, X), reshape(log(gmm.w), 1, length(gmm.w))), 2), 2)

# score_samples is underspecified by the scikit-learn API, so we're more or less free to return what we want
# http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GMM.html#sklearn.mixture.GMM
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
"""    score_samples(gmm::GMM, X::Matrix)
Return the per-sample likelihood of the data under the model. """
score_samples(gmm::GMM, X) = density(gmm, X)
