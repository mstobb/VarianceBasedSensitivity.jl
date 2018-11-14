# List of possible test functions to use

# The Sobol G Function

function sobolG(x::Array{Float64,1}; a = [0.0, 0.0, 0.0, 0.4, 0.4, 5.0])
    if length(x) != length(a)
        error("Inputs are of different lenths!")
    end

    return prod((abs.(4*x - 2) .+ a)./(1 .+ a));
end

function sobolG_analytic(; a = [0.0, 0.0, 0.0, 0.4, 0.4, 5.0])
    y = zeros(2,length(a))

    # Compute Vi, the first order indicies
    Vi = (1/3).*(1 .+a).^(-2)
    y[1,:] = Vi

    # Use the Vi to compute the total order index, but first determine all
    # possible combinations of variables:
    I = monomialDegrees(length(a),length(a))
    I[find(I.>0)] = 1
    I = unique(I,1)          # All unique combinations
    J = I[find(I[:,1] .== 1),:]    # First element is non-zero

    # Find the total variance
    Vn = 0;
    for k = 2:size(I,1) # First element is all zero, so skip
        Vn += prod(Vi[I[k,:].==1]);
    end

    for j = 1:length(a)
        # move the ith element to the start
        tempVi = [Vi[j]; Vi]
        deleteat!(tempVi,j+1)
        println(tempVi)
        for k = 1:size(J,1)
            y[2,j] += prod(tempVi[J[k,:].==1])
        end
    end

    # Devide all the computed quantities by the total variance and return
    y = y./Vn
end



function monomialDegrees(numVars::Integer, maxDegree::Integer)
## Construct the Graded polynomial index
#               I = monomialDegrees(3,2)
#
#               I =
#
#                    0     0     0
#                    1     0     0
#                    0     1     0
#                    0     0     1
#                    2     0     0
#                    1     1     0
#                    1     0     1
#                    0     2     0
#                    0     1     1
#                    0     0     2
#

# Check for the trivial case
    if numVars==1
        	degrees = collect(0:maxDegree);
    else
    	# Define needed variables
    	degrees = zeros(Int64,1,numVars);
    	k = numVars;

    	# For each degree (starting at 1) up to maxDegree...
    	for n = 1:maxDegree
    	    # Find the integers that divide up the RV permutations
    	    dividers = flipdim(transpose(hcat(collect(combinations(collect(1:(n+k-1)), k-1))...)),1);
    	    div_diff = zeros(Int64,size(dividers,1),size(dividers,2)-1);
    	    for i = 1:size(dividers,1)
    		          div_diff[i,:] = diff(vec(dividers[i,:]));
    	    end

    	    # Form the index
    	    degrees = vcat(degrees, hcat(dividers[:,1] - 1, div_diff -1, (n+k)-dividers[:,end]-1));
    	end
    	# Output result
    	degrees
    end
end
