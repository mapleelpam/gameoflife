/*
A simple example of using the gfx library.
CSE 20211
9/7/2011
by Prof. Thain
*/

#include <stdio.h>
#include <stdlib.h>
#include "gfx.h"

void draw( const int *in, int sizeOfDim )
{
	int posx, posy;
	for( posx = 0 ; posx < sizeOfDim ; posx ++ ) {
		for( posy = 0 ; posy < sizeOfDim ; posy ++ ) {
			int index = posy*sizeOfDim+posx;
			if( in[index] ) {
				gfx_color(200,200,100);
			} else {
				gfx_color(0,0,0);
			}	
			gfx_point(posx,posy);
		}
	}
}

void next(const int *in, 
		int *out, 
		int sizeOfDim 
			)
{
    // get index of the work item
	int posx, posy;

	for( posx = 0 ; posx < sizeOfDim ; posx ++ ) {
		for( posy = 0 ; posy < sizeOfDim ; posy ++ ) {
			// *begin* num of neighbor
			int num = 0;	
			if( posy > 0 ) {
				if( in[ (posy-1)*sizeOfDim+posx ])	num ++;
				if( posx > 0 && in[ (posy-1)*sizeOfDim+(posx-1) ])	num ++;
				if( posx+1 < sizeOfDim && in[ (posy-1)*sizeOfDim+(posx+1) ])	num ++;
			}
			if( posy+1 < sizeOfDim ) {
				if( in[ (posy+1)*sizeOfDim+posx ])	num ++;
				if( posx > 0 && in[ (posy+1)*sizeOfDim+(posx-1) ])	num ++;
				if( posx+1 < sizeOfDim && in[ (posy+1)*sizeOfDim+(posx+1) ])	num ++;
			}
			if( posx > 0 && in[ (posy)*sizeOfDim+(posx-1) ])	num ++;
			if( posx+1 < sizeOfDim && in[ (posy)*sizeOfDim+(posx+1) ])	num ++;
			// *end* num of neighbor
			int index = posy*sizeOfDim+posx;
    			out[index] = ( num == 3 ) || ( num == 2 && in[index] ); 
		}
	}

}


int main()
{
	int sizeOfDim = 256;
	int ysize = sizeOfDim;
	int xsize = sizeOfDim;
	int idx = 0;
	int current = 0;

	char c;

	int*	board[2];
	for(idx = 0 ; idx < 2 ; idx ++ ) {
		board[idx] = (int*) malloc(sizeof(int)*sizeOfDim*sizeOfDim);
	}
	for( idx = 0 ; idx < sizeOfDim*sizeOfDim ; idx ++ ) {
		board[0][idx] = (rand()%4) ? 0 : 1;
	}

	

	// Open a new window for drawing.
	gfx_open(xsize,ysize,"CPU");

	idx = 0;
	char str[30];
	while(1) {
		
		if( current ) {
			draw(board[1],sizeOfDim);
			next( board[1], board[0], sizeOfDim );
		} else {
			draw(board[0],sizeOfDim);
			next( board[0], board[1], sizeOfDim );
		}

		printf("CPU[%d]\n",idx++);
//		gfx_set_title(str);
		current = (current) ? 0 : 1;
	}

	return 0;
}
