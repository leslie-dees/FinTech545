using LinearAlgebra

# Define a sample matrix
n=500
sigma = fill(0.9,(n,n))
for i in 1:n
sigma[i,i]=1.0
end
sigma[1,2] = 0.7357
sigma[2,1] = 0.7357

# Function to check if a matrix is PSD
function is_psd(M)
    eigvals_M = eigen(M).values
    return all(eigvals_M .>= 0.0)
end

#Near PSD Matrix
function near_psd(a; epsilon=0.0)
    n = size(a,1)

    invSD = nothing
    out = copy(a)

    #calculate the correlation matrix if we got a covariance
    if count(x->x ≈ 1.0,diag(out)) != n
        invSD = diagm(1 ./ sqrt.(diag(out)))
        out = invSD * out * invSD
    end

    #SVD, update the eigen value and scale
    vals, vecs = eigen(out)
    vals = max.(vals,epsilon)
    T = 1 ./ (vecs .* vecs * vals)
    T = diagm(sqrt.(T))
    l = diagm(sqrt.(vals))
    B = T*vecs*l
    out = B*B'

    #Add back the variance
    if invSD !== nothing 
        invSD = diagm(1 ./ diag(invSD))
        out = invSD * out * invSD
    end
    return out
end

# Check if sigma is PSD before applying near_psd
println("Is sigma PSD before applying near_psd: ", is_psd(sigma))

# Apply the near_psd function
sigma_near_psd = near_psd(sigma)

# Check if sigma_near_psd is PSD after applying near_psd
println("Is sigma_near_psd PSD after applying near_psd: ", is_psd(sigma_near_psd))