# Contains all the types definitions

using Distributions

abstract type SobolEvals end

struct SobolDists
    P::Int
    dist::Array{Distributions.Distribution,1}
    function SobolDists(P,dist)
        length(dist) != P ?
        error("Dimensions are incorrect!") :
        new(P,dist)
    end
end


mutable struct SobolSamples
    N::Int
    P::Int
    dist::Array{Distributions.Distribution,1}
    A::Array{Float64,2}
    B::Array{Float64,2}
    function SobolSamples(N,P,dist,A,B)
        (size(A) != (N,P) || size(B) != (N,P) || length(dist) != P) ?
        error("Dimensions are incorrect!") :
        new(N,P,dist,A,B)
    end
end


mutable struct SobolEvalsSingle <: SobolEvals
    N::Int
    P::Int
    dist::Array{Distributions.Distribution,1}
    fA::Array{Float64,2}
    fB::Array{Float64,2}
    fC::Array{Array{Float64,2}}
    function SobolEvalsSingle(N,P,dist,fA,fB,fC)
        (size(fA,1) != N || size(fB,1) != N || size(fC,1) != P) ?
        error("Function evaluation numbers don't match!") :
        new(N,P,dist,fA,fB,fC)
    end
end


mutable struct SobolEvalsJoint <: SobolEvals
    N::Int
    P::Int
    dist::Array{Distributions.Distribution,1}
    Cpairs::Array{Array{Int,1}}
    fC::Array{Array{Float64,2}}
    function SobolEvalsJoint(N,P,dist,Cpairs,fC)
        (size(fC,1) != binomial(P,2)) ?
        error("Function evaluation numbers don't match!") :
        new(N,P,dist,Cpairs,fC)
    end
end


mutable struct SobolEvalsSingleAndJoint <: SobolEvals
    N::Int
    P::Int
    dist::Array{Distributions.Distribution,1}
    fA::Array{Float64,2}
    fB::Array{Float64,2}
    fC::Array{Array{Float64,2}}
    Cpairs::Array{Array{Int,1}}
    fCij::Array{Array{Float64,2}}
    function SobolEvalsSingleAndJoint(N,P,dist,fA,fB,fC,Cpairs,fCij)
        (size(fA,1) != N || size(fB,1) != N || size(fC,1) != P || size(fCij,1) != binomial(P,2)) ?
        error("Function evaluation numbers don't match!") :
        new(N,P,dist,fA,fB,fC,Cpairs,fCij)
    end
end
