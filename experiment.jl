using LinearAlgebra
using Distributions

# 定数周り
global const TestSize = 2^15                        # テストに用いるサンプル数
global const RealNumber = Float64
global const D = 3                                  # 次元
global const boundary = (-1.0, 1.0)                 # 境界
global const boundaries = [boundary for _ in 1:D]

# 生成器
function generator()::RealNumber
    res::RealNumber=0
    while true
        res = rand(Normal())
        if boundary[1] <= res && res <= boundary[2]
            break
        end
    end
    res
end
generator(d::Int64)::Array{RealNumber} = [generator() for i in 1:d]
generator(n::Int64, d::Int64)::Array{Array{RealNumber}} = [generator(d) for i in 1:n]

global const TestCase = generator(TestSize, D)

# 交差検証的な誤差評価 
#     引数   
#          f₀  :  真のパラメータ
#          f   :  推定したパラメータ 
global const epsilon = 0.0001

# 近似 Fisher Divergence
@inline function eval_error(f₀, f̂)::RealNumber
    res::RealNumber = 0
    @simd for x in TestCase
        y = copy(x)
        @simd for i in 1:D
            @inbounds y[i] += epsilon
            v::RealNumber = f₀(y) - f₀(x) - f̂(y) + f̂(x)
            res += v * v
            @inbounds y[i] -= epsilon
        end
    end
    res / (2 * TestSize * epsilon * epsilon)
end

# 推定量
@inline function estimator(
        λ::RealNumber,
        n::Int64,
        X::Array{Array{RealNumber}},
        weight_f::Array{Function},
        ∂ᵢ_weight_f::Array{Function},
        ∂ᵢ_kernel_f::Array{Function},
        ∂ᵢ²_kernel_f::Array{Function},
        ∂ᵢ∂ⱼ_kernel_f::Array{Array{Function}},
        ∂ᵢ∂ⱼ²_kernel_f::Array{Array{Function}},
        ∂ᵢ_log_q0_f::Array{Function}
    )::Function
    G::Array{ RealNumber, 2} = zeros(n * D, n * D)
    v::Array{ RealNumber, 1 } = zeros(n * D)
    idx::Int64 = 1
    nλ ::RealNumber = n * λ
    # G の計算
    @simd for b in 1:n
        @simd for j in 1:D
            @simd for a in 1:n
                @simd for i in 1:D
                    @inbounds G[idx] = sqrt( weight_f[i](X[a]) * weight_f[j](X[b]) ) * ∂ᵢ∂ⱼ_kernel_f[i][j](X[a], X[b])
                    idx += 1
                end
            end
        end
    end
    idx = 1
    @simd for a in 1:n
        @simd for i in 1:D
            @inbounds G[idx, idx] += nλ
            idx += 1
        end
    end
    # v の計算
    idx = 1
    @simd for a in 1:n
        @simd for i in 1:D
            @simd for b in 1:n
                @simd for j in 1:D
                    @inbounds v[idx] += weight_f[j](X[b]) * ∂ᵢ∂ⱼ²_kernel_f[i][j](X[a], X[b])
                    @inbounds v[idx] += ( weight_f[j](X[b]) * ∂ᵢ_log_q0_f[j](X[b]) + ∂ᵢ_weight_f[j](X[b]) ) * ∂ᵢ∂ⱼ_kernel_f[i][j](X[a], X[b])
                end
            end
            v[idx] *= sqrt(weight_f[i](X[a]))
            v[idx] /= nλ
            idx += 1
        end
    end

    β::Array{RealNumber, 1} = G \ v
    idx = 1
    @simd for a in 1:n
        @simd for i in 1:D
            @inbounds β[idx] *= sqrt( weight_f[i](X[a]) )
            idx += 1
        end
    end

    # 推定量(パラメータ関数)
    @inline function estimator_function(x::Array{RealNumber})::RealNumber
    local value1::RealNumber = 0
    local value2::RealNumber = 0
    local index::Int64 = 1
        @simd for a in 1:n
            @simd for i in 1:D
                @inbounds value1 += weight_f[i](X[a]) * ∂ᵢ²_kernel_f[i](X[a], x)
                @inbounds value1 += (weight_f[i](X[a]) * ∂ᵢ_log_q0_f[i](X[a]) + ∂ᵢ_weight_f[i](X[a])) * ∂ᵢ_kernel_f[i](X[a], x)
                @inbounds value2 += β[index] * ∂ᵢ_kernel_f[i](X[a], x)
                index += 1
            end
        end
        (- value1 / nλ) + value2
    end
    estimator_function
end

# 推定量 : 計算量 O(n^3d) (行列定義込)
@inline function approximate_matching_estimator(
        λ::RealNumber,
        n::Int64,
        X::Array{Array{RealNumber}},
        weight_f::Array{Function},
        ∂ᵢ_weight_f::Array{Function},
        kernel_f::Function,
        ∂ᵢ_kernel_f::Array{Function},
        ∂ᵢ²_kernel_f::Array{Function},
        ∂ᵢ_log_q0_f::Array{Function}
    )::Function

    A::Array{ RealNumber, 2 } = zeros(n, n)
    b::Array{ RealNumber, 1 } = zeros(n)
    @simd for j in 1:n
        @simd for i in 1:n
            @simd for l in 1:D
                @simd for a in 1:n
                    @inbounds A[i, j] += weight_f[l](X[a]) * ∂ᵢ_kernel_f[l](X[a], X[j]) * ∂ᵢ_kernel_f[l](X[a], X[i])
                end
            end
            @inbounds A[i, j] += n * λ * kernel_f(X[i], X[j])
        end
    end
    @simd for i in 1:n
        @simd for l in 1:D
            @simd for a in 1:n
                @inbounds b[i] += weight_f[l](X[a]) * ∂ᵢ²_kernel_f[l](X[a], X[i])
                @inbounds b[i] += ( weight_f[l](X[a]) * ∂ᵢ_log_q0_f[l](X[a]) + ∂ᵢ_weight_f[l](X[a]) ) * ∂ᵢ_kernel_f[l](X[a], X[i])
            end
        end
    end
    α::Array{RealNumber, 1} = -(A \ b)
    # 推定量(パラメータ関数)
    @inline function approximate_matching_estimator_function(x::Array{RealNumber})::RealNumber
        local res::RealNumber = 0
        @simd for i in 1:n
            res += α[i] * kernel_f(x, X[i])
        end
        res
    end
    approximate_matching_estimator_function
end

# 重み関数とカーネル
@inline function weight_func(i::Int)::Function
    (x::Array{RealNumber}) -> cbrt(min(x[i] - boundaries[i][1], boundaries[i][2] - x[i])) #-(x[i] - boundaries[i][1]) * (x[i] - boundaries[i][2])
end
@inline function ∂ᵢ_weight_func(i::Int)::Function
    (x::Array{RealNumber}) -> (x[i] < (boundaries[i][1] + boundaries[i][2]) / 2) ? 1.0 / (3 * cbrt(x[i] - boundaries[i][1])^2) : -1.0 / (3 * cbrt(boundaries[i][2] - x[i])^2) #-2x[i] + boundaries[i][1] + boundaries[i][2]
end
# カーネル
@inline function kernel_function(x::Array{RealNumber}, y::Array{RealNumber})::RealNumber
    (x ⋅ y)^2
end
@inline function ∂ᵢ_kernel_func(i::Int)::Function
    (x::Array{RealNumber}, y::Array{RealNumber}) -> 2 * y[i] * (x ⋅ y)
end
@inline function ∂ᵢ²_kernel_func(i::Int)::Function
    (x::Array{RealNumber}, y::Array{RealNumber}) -> 2 * y[i] * y[i]
end
@inline function ∂ᵢ∂ⱼ_kernel_func(i::Int, j::Int)::Function
    if i == j
        (x::Array{RealNumber}, y::Array{RealNumber}) -> 2 * (x[i] * y[i] + (x ⋅ y))
    else
        (x::Array{RealNumber}, y::Array{RealNumber}) -> 2 * x[j] * y[i]
    end
end
@inline function ∂ᵢ∂ⱼ²_kernel_func(i::Int, j::Int)::Function
    if i == j
        (x::Array{RealNumber}, y::Array{RealNumber}) -> 4 * x[i]
    else
        (x::Array{RealNumber}, y::Array{RealNumber}) -> 0
    end
end
@inline function ∂ᵢ_log_q0_func(i::Int)::Function
    (x::Array{RealNumber}) -> 0
end

# 学習用 データ
global const DataSize = [20, 40, 60, 80, 100]
global const Loop = 10
global const Data = [[generator(m, D) for j in 1:Loop] for m in DataSize]

global const weight_functions = Array{Function}([weight_func(i) for i in 1:D])
global const ∂ᵢ_weight_functions = Array{Function}([∂ᵢ_weight_func(i) for i in 1:D])
global const ∂ᵢ_kernel_functions = Array{Function}([∂ᵢ_kernel_func(i) for i in 1:D])
global const ∂ᵢ²_kernel_functions = Array{Function}([∂ᵢ²_kernel_func(i) for i in 1:D])
global const ∂ᵢ∂ⱼ_kernel_functions = Array{Array{Function}}([[∂ᵢ∂ⱼ_kernel_func(i, j) for j in 1:D] for i in 1:D])
global const ∂ᵢ∂ⱼ²_kernel_functions = Array{Array{Function}}([[∂ᵢ∂ⱼ²_kernel_func(i, j) for j in 1:D] for i in 1:D])
global const ∂ᵢ_log_q0_functions = Array{Function}([∂ᵢ_log_q0_func(i) for i in 1:D])

@inline function F(x::Array{RealNumber})::RealNumber
    - (x ⋅ x) / 2
end

@inline function eval(i::Int)::Nothing
    println("$(DataSize[i]):")
    E = zeros(RealNumber, Loop)
    λ = 1.0 / cbrt(DataSize[i])
    for t in 1:Loop
        print("        $(t) : ")

        f̂ = estimator(λ, DataSize[i], Data[i][t], 
            weight_functions, ∂ᵢ_weight_functions, 
            ∂ᵢ_kernel_functions, ∂ᵢ²_kernel_functions, ∂ᵢ∂ⱼ_kernel_functions, ∂ᵢ∂ⱼ²_kernel_functions, 
            ∂ᵢ_log_q0_functions
        )
        """
        f̂ = approximate_matching_estimator(λ, DataSize[i], Data[i][t],
            weight_functions, ∂ᵢ_weight_functions,
            kernel_function, ∂ᵢ_kernel_functions, ∂ᵢ²_kernel_functions, ∂ᵢ_log_q0_functions
        )
        """
        err::RealNumber= eval_error(F, f̂)
        println(err)
        E[t] = err
    end
    println("    -> mean : $(mean(E))    ;    [$(minimum(E)), $(maximum(E))]")
end

for i in 1:length(DataSize)
    eval(i)
end
