using Distributions
using OnlineStats


module VarianceBasedSensitivity

# Include the type definitions
include("types.jl")


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


# Include the sensitivity metric calculations
include("sobol_inds.jl")


end # module
