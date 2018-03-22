module VarianceBasedSensitivity

using Distributions
using OnlineStats

export SobolDists, SobolSamples, SobolEvals
export SobolEvalsSingle, SobolEvalsJoint
export SobolEvalsSingleAndJoint
export makeSobolSamples
export makeNewSobolSamples!
export +
export makeC, makeC!, makeCij
export sobolSampler, sobolSamplerSij
export sobolSamplerSingle
export computeSensSi, computeSensSiT
export computeSensSij
export sensSiSTD, sensSiTSTD, sensSijSTD

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


function makeSobolSamples(S::SobolDists,N::Int)
    # Form the A and B matricies
    A = zeros(Float64,N,S.P)
    B = zeros(Float64,N,S.P)
    for i = 1:S.P
        A[:,i] = Base.rand(S.dist[i],N)
        B[:,i] = Base.rand(S.dist[i],N)
    end
    return SobolSamples(N,S.P,S.dist,A,B)
end


function makeNewSobolSamples!(S::SobolSamples,N::Int)
    # Form the A and B matricies
    A = zeros(N,S.P)
    B = zeros(N,S.P)
    for i = 1:S.P
        A[:,i] = rand(S.dist[i],N)
        B[:,i] = rand(S.dist[i],N)
    end
    S.A = vcat(S.A,A)
    S.B = vcat(S.B,B)
    S.N += N
end


import Base.+
function +(x::SobolSamples,y::SobolSamples)
    # If the distributions match, add together
    if x.dist == y.dist
	z = SobolSamples((x.N+y.N),x.P,x.dist,vcat(x.A,y.A),vcat(x.B,y.B))
    else
        error("Distributions don't match!")
    end
    z
end


function +(x::SobolEvalsSingle,y::SobolEvalsSingle)
    # If the distributions match, add together
    if (x.dist == y.dist)
	tempfC = Array{Array{Float64,2}}(x.P)
        for i = 1:x.P
            tempfC[i] = vcat(x.fC[i],y.fC[i])
        end
	z = SobolEvalsSingle((x.N+y.N),x.P,x.dist,vcat(x.fA,y.fA),vcat(x.fB,y.fB),tempfC)
    else
        error("Distributions don't match!")
    end
    z
end


function +(x::SobolEvalsSingleAndJoint,y::SobolEvalsSingleAndJoint)
    # If the distributions match, add together
    if (x.dist == y.dist)
	tempfC = Array{Array{Float64,2}}(x.P)
        for i = 1:x.P
            tempfC[i] = vcat(x.fC[i],y.fC[i])
        end
	tempfCij = Array{Array{Float64,2}}(length(x.Cpairs))
        for i = 1:length(x.Cpairs)
            tempfCij[i] = vcat(x.fCij[i],y.fCij[i])
        end
	z = SobolEvalsSingleAndJoint((x.N+y.N),x.P,x.dist,vcat(x.fA,y.fA),vcat(x.fB,y.fB),tempfC,x.Cpairs,tempfCij)
    else
        error("Distributions don't match!")
    end
    z
end


function makeC(S::SobolSamples, k::Int)
    C = deepcopy(S.B)
    for i = 1:S.N
        C[i,k] = S.A[i,k];
    end
    C
end


function makeC(S::SobolSamples, i::Int, k::Int)
    C = deepcopy(S.B[k,:])
    C[i] = S.A[k,i];
    C
end


function makeC!(C::Array{Float64,2},S::SobolSamples, k::Int)
    if size(C) == size(S.A)
        for i = 1:S.N, j = vcat(collect(1:k-1),collect(k+1:S.P))
            C[i,j] = S.B[i,j]
        end
        for i = 1:S.N
            C[i,k] = S.A[i,k];
        end
    else
        error("Given C is the wrong sie!")
    end
end


function makeCij(S::SobolSamples, i::Int, j::Int, k::Int)
    C = deepcopy(S.B[k,:])
    C[i] = S.A[k,i]
    C[j] = S.A[k,j]
    C
end



function sobolSamplerSingle(f::Function,sob::SobolSamples)

    fout = f(sob.A[1,:])

    #sample_results = zeros(Float64,sob.N*(sob.P + 2),length(fout))
    sample_results = SharedArray{Float64,2}((sob.N*(sob.P + 2),length(fout)), init = 0)

    @sync @parallel for i = 1:size(sample_results,1)
        if i < sob.N + 1
            sample_results[i,:] = f(sob.A[i,:])

        elseif  i < 2*sob.N + 1
            sample_results[i,:] = f(sob.B[i-sob.N,:])

        else
            n = Int(mod(i-2*sob.N,sob.N))
            n == 0 ? n = sob.N : n
            cval = Int(floor((i - 2*sob.N)/sob.N) + 1)
            n == sob.N ? cval = cval - 1 : cval
            #println("i = ",i,", n = ",n,", cval = ",cval)
            sample_results[i,:] = f(makeC(sob,cval,n))
        end
    end

    formC = Array{Array{Float64,2}}(sob.P)
    for i = 1:length(formC)
        formC[i] = sample_results[(2+i-1)*sob.N+1:(2+i)*sob.N,:]
    end

    #println(sample_results)
    println(sample_results[1:10,:])
    println("  ")

    SobolEvalsSingle(sob.N,sob.P,sob.dist,sample_results[1:sob.N,:],sample_results[sob.N+1:2*sob.N,:],formC)

end


function sobolSamplerJoint(f::Function,sob::SobolSamples)

    fout = f(sob.A[1,:])

    sample_results = zeros(binomial(sob.P,2)*sob.N,length(fout))
    formctemp = zeros(Int,binomial(sob.P,2),2)
    counter = 1
    for i = 1:sob.P-1, j = i+1:sob.P
        formctemp[counter,:] = [i,j]
        counter += 1
    end
    Cij = kron(formctemp, ones(Int,sob.N))


    for i = 1:size(sample_results,1)
            n = Int(mod(i,sob.N))
            n == 0 ? n = sob.N : n
            #println("i = ",i,", n = ",n,", cval = ",cval)
            sample_results[i,:] = f(makeCij(sob,Cij[i,1],Cij[i,2],n))
    end

    CijDict = Array{Array{Int64,1}}(binomial(sob.P,2))
    formCij = Array{Array{Float64,2}}(binomial(sob.P,2))
    for i = 1:length(formCij)
        CijDict[i] = formctemp[i,:]
        formCij[i] = sample_results[(i-1)*sob.N+1:(i)*sob.N,:]
    end

    SobolEvalsJoint(sob.N,sob.P,sob.dist,CijDict,formCij)

end

function singleSobolSampleEvaluator(i::Int,output::SharedArray{Float64,2},sob::SobolSamples,f::Function)
	if i < sob.N + 1
		output[i,:] = f(sob.A[i,:])

	elseif  i < 2*sob.N + 1
		output[i,:] = f(sob.B[i-sob.N,:])

	else
		n = Int(mod(i-2*sob.N,sob.N))
		n == 0 ? n = sob.N : n
		cval = Int(floor((i - 2*sob.N)/sob.N) + 1)
		n == sob.N ? cval = cval - 1 : cval
		output[i,:] = f(makeC(sob,cval,n))
	end
end

function jointSobolSampleEvaluator(i::Int,output::SharedArray{Float64,2},Cij::Array{Int64,2},sob::SobolSamples,f::Function)
	n = Int(mod(i,sob.N))
        n == 0 ? n = sob.N : n
        output[i,:] = f(makeCij(sob,Cij[i,1],Cij[i,2],n))
end


function sobolSampler(f::Function,sob::SobolSamples)

    fout = f(sob.A[1,:])

    #sample_results = zeros(sob.N*(sob.P + 2),length(fout))
    sample_results = SharedArray{Float64,2}((sob.N*(sob.P + 2),length(fout)), init = 0)

    # my pmap call is not working correctly and is FAR too slow
    #pmap((x)->singleSobolSampleEvaluator(x,sample_results,sob,f),1:sob.N*(sob.P+2); batch_size = 100)
    @sync @parallel for i = 1:size(sample_results,1)
        singleSobolSampleEvaluator(i,sample_results,sob,f)
    end

    formC = Array{Array{Float64,2}}(sob.P)
    for i = 1:length(formC)
        formC[i] = sample_results[(2+i-1)*sob.N+1:(2+i)*sob.N,:]
    end

    fA = sample_results[1:sob.N,:]
    fB = sample_results[sob.N+1:2*sob.N,:]

    #sample_results = zeros(binomial(sob.P,2)*sob.N,length(fout))
    sample_results = SharedArray{Float64,2}((binomial(sob.P,2)*sob.N,length(fout)), init = 0)

    formctemp = zeros(Int,binomial(sob.P,2),2)
    counter = 1
    for i = 1:sob.P-1, j = i+1:sob.P
        formctemp[counter,:] = [i,j]
        counter += 1
    end
    Cij = kron(formctemp, ones(Int,sob.N))

    # pmap((x)->jointSobolSampleEvaluator(x,sample_results,Cij,sob,f),1:(binomial(sob.P,2)*sob.N); batch_size = 100)
    @sync @parallel for i = 1:size(sample_results,1)
        jointSobolSampleEvaluator(i,sample_results,Cij,sob,f)
    end

    CijDict = Array{Array{Int64,1}}(binomial(sob.P,2))
    formCij = Array{Array{Float64,2}}(binomial(sob.P,2))
    for i = 1:length(formCij)
        CijDict[i] = formctemp[i,:]
        formCij[i] = sample_results[(i-1)*sob.N+1:(i)*sob.N,:]
    end

    SobolEvalsSingleAndJoint(sob.N,sob.P,sob.dist,fA,fB,formC,CijDict,formCij)

end


function computeSensSi(S::SobolEvals)

    sensSi = zeros(S.P, size(S.fA,2))

    vy = var(S.fA,1)

    for i = 1:S.P
        vari = sum(S.fA .* S.fC[i],1)./(S.N-1.0)
        mmi = mean(S.fA,1).*mean(S.fC[i],1)
        sensSi[i,:] = (vari.-mmi)./vy
    end

    sensSi
end


function computeSensSiT(S::SobolEvals)

    sensSiT = zeros(S.P, size(S.fA,2))

    vy = var(S.fA,1)

    for i = 1:S.P
        vari = (1/(2*S.N)).*sum((S.fB .- S.fC[i]).^2,1)
        sensSiT[i,:] = vari./vy
    end

    sensSiT
end


function computeSensSiboot!(sensSi::Array{Float64,2},S::SobolEvals,inds)

    vy = var(S.fA[inds,:],1)

    for i = 1:S.P
        vari = sum(S.fA[inds,:] .* S.fC[i][inds,:],1)
        mmi = mean(S.fA[inds,:],1).* mean(S.fC[i][inds,:],1)
        sensSi[i,:] = (vari/(S.N-1.0)-mmi)./vy
    end

    sensSi
end


function computeSensSiTboot!(sensSiT::Array{Float64,2},S::SobolEvals,inds)

    vy = var(S.fA[inds,:],1)

    for i = 1:S.P
        vari = (1/(2*S.N)).*sum((S.fB[inds,:] .- S.fC[i][inds,:]).^2,1)
        sensSiT[i,:] = vari./vy
    end

    sensSiT
end




function computeSensSij(S::SobolEvalsSingleAndJoint)

    sensSi = zeros(S.P, size(S.fA,2))
    sensSij = zeros(length(S.Cpairs), size(S.fA,2))

    vy = var(S.fA,1)

    for i = 1:S.P
        vari = sum(S.fA .* S.fC[i],1)
        mmi = mean(S.fA,1).* mean(S.fC[i],1)
        sensSi[i,:] = (vari/(S.N-1.0)-mmi)./vy
    end

    for i = 1:length(S.Cpairs)
        vari = sum(S.fA .* S.fCij[i],1)
        mmi = mean(S.fA,1).* mean(S.fCij[i],1)
        sensSij[i,:] = (vari/(S.N-1.0)-mmi)./vy
    end
    for i = 1:length(S.Cpairs)
        sensSij[i,:] -= (sensSi[S.Cpairs[i][1],:] .+ sensSi[S.Cpairs[i][2],:])
    end

    sensSi, sensSij
end


function computeSensSijT(S::SobolEvalsSingleAndJoint)

    sensSiT = zeros(S.P, size(S.fA,2))

    vy = var(S.fA,1)

    for i = 1:S.P
        vari = (1/(2*S.N)).*sum((S.fB .- S.fC[i]).^2,1)
        sensSiT[i,:] = vari./vy
    end

    sensSiT
end


function computeSensSijboot!(sensSi::Array{Float64,2},sensSij::Array{Float64,2},S::SobolEvalsSingleAndJoint,inds)

    vy = var(S.fA[inds,:],1)

    for i = 1:S.P
        vari = sum(view(S.fA,inds,:) .* view(S.fC[i],inds,:),1)
        mmi = mean(view(S.fA,inds,:),1).* mean(view(S.fC[i],inds,:),1)
        sensSi[i,:] = (vari/(S.N-1.0)-mmi)./vy
    end

    for i = 1:length(S.Cpairs)
        vari = sum(view(S.fA,inds,:) .* view(S.fCij[i],inds,:),1)
        mmi = mean(view(S.fA,inds,:),1).* mean(view(S.fCij[i],inds,:),1)
        sensSij[i,:] = (vari/(S.N-1.0)-mmi)./vy
    end
    for i = 1:length(S.Cpairs)
        sensSij[i,:] -= (sensSi[S.Cpairs[i][1],:] .+ sensSi[S.Cpairs[i][2],:])
    end

    sensSi, sensSij
end


function computeSensSijTboot!(sensSiT::Array{Float64,2},S::SobolEvalsSingleAndJoint,inds)

    vy = var(S.fA[inds,:],1)

    for i = 1:S.P
        vari = (1/(2*S.N)).*sum((S.fB[inds,:] .- S.fC[i][inds,:]).^2,1)
        sensSiT[i,:] = vari./vy
    end

    sensSiT
end


function sensSijSTD(S::SobolEvalsSingleAndJoint, N::Int)

    sensSij = computeSensSij(S)
    sensSTD = Array{OnlineStats.Series,2}(size(sensSij[1]))
    apxhist = Array{OnlineStats.Hist,2}(size(sensSij[1]))
    sensSijSTD = Array{OnlineStats.Series,2}(size(sensSij[2]))
    apxhistSij = Array{OnlineStats.Hist,2}(size(sensSij[2]))
    for i = 1:size(sensSij[1],1), j = 1:size(sensSij[1],2)
        apxhist[i,j] = Hist(-0.1:0.01:1.1)
        sensSTD[i,j] = Series(sensSij[1][i,j], apxhist[i,j], Mean(), Variance(), Extrema())
    end
    for i = 1:size(sensSij[2],1), j = 1:size(sensSij[2],2)
        apxhistSij[i,j] = Hist(-0.1:0.01:1.1)
        sensSijSTD[i,j] = Series(sensSij[2][i,j], apxhistSij[i,j], Mean(), Variance(), Extrema())
    end

    for i = 2:N
        samp = rand(1:S.N,S.N)
        computeSensSijboot!(sensSij[1],sensSij[2],S,samp)
        for i = 1:size(sensSij[1],1), j = 1:size(sensSij[1],2)
            fit!(sensSTD[i,j],sensSij[1][i,j])
        end
	for i = 1:size(sensSij[2],1), j = 1:size(sensSij[2],2)
            fit!(sensSijSTD[i,j],sensSij[2][i,j])
        end
    end
    sensSTD,sensSijSTD
end



function sensSiSTD(S::SobolEvals, N::Int)

    sensSi = computeSensSi(S)
    sensSTD = Array{OnlineStats.Series,2}(size(sensSi))
    apxhist = Array{OnlineStats.Hist,2}(size(sensSi))
    for i = 1:size(sensSi,1), j = 1:size(sensSi,2)
        apxhist[i,j] = Hist(-0.1:0.001:1.1)
        sensSTD[i,j] = Series(sensSi[i,j], apxhist[i,j], Mean(), Variance(), Extrema())
    end

    for i = 2:N
        samp = rand(1:S.N,S.N)
        computeSensSiboot!(sensSi,S,samp)
        for i = 1:size(sensSi,1), j = 1:size(sensSi,2)
            fit!(sensSTD[i,j],sensSi[i,j])
        end
    end
    sensSTD
end


function sensSiTSTD(S::SobolEvals, N::Int)

    sensSiT = computeSensSiT(S)
    sensSTD = Array{OnlineStats.Series,2}(size(sensSiT))
    apxhist = Array{OnlineStats.Hist,2}(size(sensSiT))
    for i = 1:size(sensSiT,1), j = 1:size(sensSiT,2)
        apxhist[i,j] = Hist(-0.1:0.001:1.1)
        sensSTD[i,j] = Series(sensSiT[i,j], apxhist[i,j], Mean(), Variance(), Extrema())
    end

    for i = 2:N
        samp = rand(1:S.N,S.N)
        computeSensSiTboot!(sensSiT,S,samp)
        for i = 1:size(sensSiT,1), j = 1:size(sensSiT,2)
            fit!(sensSTD[i,j],sensSiT[i,j])
        end
    end
    sensSTD
end


end # module
