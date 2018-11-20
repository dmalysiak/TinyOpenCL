__kernel void tiny(__global unsigned int* A)
{
	int idx = get_global_id(0);
	
	A[idx] = idx;

}