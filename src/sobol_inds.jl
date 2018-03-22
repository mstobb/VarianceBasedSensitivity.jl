# Code to compute the actual sensitivity indices

using OnlineStats

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
