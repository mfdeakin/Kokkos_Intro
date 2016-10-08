
#include <Kokkos_Core.hpp>

template <typename real>
struct SquareReduce {
	KOKKOS_INLINE_FUNCTION
	void operator() (const real i, real &lsum) const {
		lsum += i * i;
	}
};

int main(int argc, char **argv) {
	Kokkos::initialize(argc, argv);
	
	using real = int;
	constexpr const int n = 100000;
	real sum = 0;
	Kokkos::parallel_reduce(n, SquareReduce<real> (), sum);
	printf("Sum of squares from 1 to %d: %d\n", n - 1, sum);
	real check_sum = 0;
	for(real i = 1; i < n; i++) {
		check_sum += i * i;
	}
	printf("check_sum: %d\n", check_sum);
	Kokkos::finalize();
	return 0;
}
