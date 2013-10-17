 // ACL kernel for adding two input vectors

int numOfNeighbor( __global const int *matrix, int posx, int posy, int sizeOfDim )
{
    // *begin* num of neighbor
    int num = 0;	
    if( posy > 0 ) {
	if( matrix[ (posy-1)*sizeOfDim+posx ])	num ++;
	if( posx > 0 && matrix[ (posy-1)*sizeOfDim+(posx-1) ])	num ++;
	if( posx+1 < sizeOfDim && matrix[ (posy-1)*sizeOfDim+(posx+1) ])	num ++;
    }
    if( posy+1 < sizeOfDim ) {
	if( matrix[ (posy+1)*sizeOfDim+posx ])	num ++;
	if( posx > 0 && matrix[ (posy+1)*sizeOfDim+(posx-1) ])	num ++;
	if( posx+1 < sizeOfDim && matrix[ (posy+1)*sizeOfDim+(posx+1) ])	num ++;
    }
    if( posx > 0 && matrix[ (posy)*sizeOfDim+(posx-1) ])	num ++;
    if( posx+1 < sizeOfDim && matrix[ (posy)*sizeOfDim+(posx+1) ])	num ++;
    // *end* num of neighbor

    return num;
} 

__kernel void next(__global const int *in, 
                        __global int *restrict out, 
                        int sizeOfDim 
			)
{
    // get index of the work item
    int index = get_global_id(0);
    int posx = index % sizeOfDim;
    int posy = index / sizeOfDim; 

    int num = numOfNeighbor( in, posx, posy, sizeOfDim);

    out[index] = ( num == 3 ) || ( num == 2 && in[index] ); 
}
