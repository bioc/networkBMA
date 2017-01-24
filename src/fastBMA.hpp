#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <set>
#include <map>
#include <limits>
#include "my_sort.hpp"
#include <unordered_set>
#define USE_RBLAS //for now always use the default RBLAS library
#ifdef _OPENMP
 #include <omp.h>
#endif

#include <inttypes.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <boost/math/tools/minima.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stdio.h>
#include <bitset>
#include "MurmurHash3.h"
#ifdef USEMPI
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/string.hpp>
namespace mpi = boost::mpi;
#endif

#ifdef USE_RBLAS
 #include <blas.h> //this must go AFTER and not before the boost mpi headers for some reason
#else 
 #include <cblas.h> //this must go AFTER and not before the boost mpi headers for some reason
#endif

#include <lapacke.h>


#define UNIFORM_PRIOR 2.76/6000.0
#define EPSILONFACTOR 1e-6 //will be added to uniform prior if correlation matrix is given to casue the variables to be ranked
#define MAXPRIOR 0.9999999 //needed for stability
#define NFILTERS 64

using namespace std;


//globals

//unsigned int total_checks=0;
//unsigned int total_collisions=0;

uint32_t gMaxKeptModels=100000;
uint32_t gMaxActiveModels=100000;


uint32_t *hashLUT=0; //hash lookup table
uint8_t gModelBits=1; //number of bits needed to represent nModels

//function to initialize hashLUT
	//precalculate hashes
void initHashLUT(int nVars){
	if(hashLUT) delete[] hashLUT;
	hashLUT=new uint32_t[nVars];
	const int32_t seed=0xfaf6cdb3;
	for (int i=0;i<nVars;i++){
		uint32_t h=i;
		MurmurHash3_x86_32(&h,4,seed,hashLUT+i);
	}
}	
void findModelBits(int nVars){
	uint8_t nBits=1;
	uint64_t bitValue=2;
	while((uint64_t)nVars >  bitValue){
		bitValue*=2;
		nBits++;
	}
	gModelBits=nBits;
}	
	
//set up for timeSeries
template <class T> T TimeSeriesValuesToResiduals(T *values, T *residuals,int nTimes,int nGroups);


//class definitions
//DenseTrMatrix stores an upper triangular matrix to save room for storage of models
//The Indices classes take advantage of the fact that our set of variables is always limited in size and that this is known beforehand
//So at a minor cost in memory we can avoid a linked list
//Instead we have a list that stores the indices and an index array that points to the location in the list where an index resides
//ie. list[index[model]-1]=model (zero is reserved to indicate that a model is not in the list so we add 1 to the locations when we store them in the index array)
//The basic implementation is the ModelIndices class for this
//CompactModelIndices eliminates the index altogether. Presently we don't need random access to the list except during insert deletion so this saves the memory cost of the index - important for the hash which keeps track of all the modelsets that have been evaluated
//A bitArray index is generated when needed to compare two sets of indices as needed
//memory could be further reduced by templating the indices to allow for char and short unsigned int indices (256 and 64K limit) or using a bitarray to store the elements which would also be faster. Alternatively use a suffix array to store the model sets similar to what is done for short read sequencing. For maintaining a list of very large sets  that would probably be the best way to go instead of the current hash system


//some definitions needed for Dijkstra
//add early abort if the route is already worse than the existing route
class Comparator{
 public:
 int operator() ( const pair<int,float>& p1, const pair<int,float> &p2)
 {
 return p1.second>p2.second;
 }
};
class BitArray{
	public:	
	unsigned char *array=0;
 size_t nBytes;
	BitArray(size_t nBits){
		nBytes=(nBits%8) ? nBits/8+1: nBits/8;
		array=new unsigned char[nBytes];
		memset(array,0,nBytes);
	}
	BitArray(const BitArray &A){
 	nBytes=A.nBytes;
 	if(array){delete[] array;array=0;}
		array=new unsigned char[nBytes];
		memmove(array,A.array,nBytes);
	}
	unsigned char getBit(size_t index) {
  return (array[index/8] >> 7-(index & 0x7)) & 0x1;
 }
 unsigned char setBit(size_t index) {
  array[index/8] = array[index/8] |  1 << 7-(index & 0x7);
  return(1);
 }
 unsigned char getSetBit(size_t index) {
		const unsigned char retValue=array[index/8] >> (7-(index & 0x7)) & 0x1;
		if(!retValue)array[index/8] = array[index/8] |  1 << (7-(index & 0x7));
		return(retValue);
 }
 void clear(){
		memset(array,0,nBytes);
	}
  ~BitArray(){
  if(array)delete [] array; 
	}
};
template <class T> class PackedBitArray{
	//T is an UNSIGNED int type
	//returns a 64 bit returnvalue - this is necessary for the algorithm to work endian independently 
	//if the number of bits return value is less than size of T - then you need to worry about endianness in constructing the final output
	//T of uint64_t is the fastest because of fewer word overlaps and the extra size is minimal - probably stores 64 bit offset arrays anyway
	public:
	T *array=0;
	PackedBitArray(){
		array=0;
	}	
	PackedBitArray(uint8_t elementSize,size_t nElements){
		if(array)delete[] array;
	 const size_t nBits=nElements*elementSize;
	 const uint8_t tBits=sizeof(T) *8;
		size_t nWords=(nBits&(tBits-1)) ? (nBits/tBits)+1: nBits/tBits;
		array=new T [nWords];
		memset(array,0,nWords*sizeof(T));
	}
 uint64_t get (uint8_t elementSize,size_t n) {
	 //want nth element
	 const uint8_t tBits=sizeof(T) *8;
		size_t index=n*elementSize;
		size_t startIndex=index/tBits;
		size_t endIndex=(index+elementSize-1)/tBits;
	 uint8_t bitOffset=index & (tBits-1);
		
		//case 1 start and end Index is same - just shift	
		if(startIndex==endIndex){
			//careful if you do it all in the same expression it does always not lose the bits - maybe it keeps it in a 64 bit or 32 bit format natively
			//shift bits left to zero left bits and shift right to correct possition
			//example for 3 bits
			//aaabbbcc -> bbbcc000 ->00000bbb
		 T lshift=(array[startIndex] << bitOffset);
			return(lshift >> (tBits-elementSize)); 
		}
		//if it is over several words
	 //eg for case of 13 bits and T unint32
	 //aaaaaaaa aaaaaxxx xxxxxxxx xxbbbbbb
  //read leading bits
  uint8_t bitsRead=tBits-bitOffset;
  T ones=~0;
	 uint64_t retValue=( ones >> bitOffset) & array[startIndex];
		//00000000 00000000 00000000 000000xxx 
		retValue=retValue << (elementSize-bitsRead); //00000000 00000000 00xxx000 00000000 
		//read internal bits	 
		for(int i=startIndex+1;i<endIndex;i++){
		 retValue |=  ((uint64_t) array[i]) << elementSize-bitsRead-tBits;
		 bitsRead+=8;
		}
		//read trailing bits
		//right shift to zero out rightmost bits	
		//xxbbbbbb -> 00000000 00000000 00000000 xxbbbbbb -> 00000000 00000000 00000000 000000xx 
	 retValue |=  ((uint64_t) array[endIndex]) >> (tBits-(elementSize-bitsRead)); 
  return retValue;
 }
 void set(uint8_t elementSize, size_t n, uint64_t value){
		const uint8_t tBits=sizeof(T) *8;
		size_t index=n*elementSize;
		size_t startIndex=index/tBits;
		size_t endIndex=(index+elementSize-1)/tBits;
	 uint8_t bitOffset=index & (tBits-1);
		//eg for case of 13 bits and T unint32
	 //aaaaaaaa aaaaaxxx xxxxxxxx xxbbbbbb
	 //fprintf(stderr,"start %d end %d offset %d\n",startIndex,endIndex,bitOffset);
		if(startIndex==endIndex){
		 //prepare mask
		 //example for elementSize  11111111 -> 00000111 -> 00111000 -> 11000111 <-desired mask
		 T mask= ~0; //can't do this in an expression or it defaults to a longer word
		 mask= mask >> (tBits-elementSize);
		 mask= mask << (tBits- bitOffset -elementSize);
   //00000xxx -> 00xxx000
		 array[startIndex]=array[startIndex] & ~mask | ((T) value << (tBits- bitOffset -elementSize));
		}
		else{
		 //example
		 //00000000 00000000 000xxxxx xxxxxxxx
		 //into 
		 //aaaaaaab bbbbbbbb bbbbcccc cccccccc
		 //leading bit mask is 11111110
		 //value is shifted right by elementSize - (8-index%8)
		 //00000000 00000000 00000000 0000000x -> cast -> 0000000x
		 //aaaaaaab & 11111110 | 0000000x = aaaaaaax
		 T ones=~0;
		 T mask= ones << (tBits-bitOffset);		
			array[startIndex]=(array[startIndex] & mask) | (T)(value >> elementSize  - (tBits-bitOffset));
			size_t bitsWritten=tBits-bitOffset;
   //internal bits is a simple shift - no mask needed when nElemet
   for(int i=startIndex+1;i<endIndex;i++){
				array[i]=(T) (value >> elementSize-bitsWritten-tBits);
				bitsWritten+=8;
			}
			//trailing bit mask is 00001111
			//shift left the value by the number of remaining bits
			//00000000 00000000 000xxxxx xxxxxxxx -> 00000000 0000000x xxxxxxxx xxxx0000 -> cast -> xxxx0000
			mask= ones >> elementSize-bitsWritten;
			array[endIndex]=array[endIndex] & mask | (T) (value << (tBits-(elementSize-bitsWritten))); 			
		}	
	}	
 ~PackedBitArray(){
  if(array)delete [] array;
	}
};	
template <class T> class  BitIndex{//T is for type of list or type of PackedBit array
	//simple class meant for throwaway comparisons
 BitArray *bitArray=0; 
 uint32_t minValue;
 uint32_t maxValue;
 size_t arraySize=0;
 public:
 BitIndex(T *list,T nModels){
		maxValue=list[0];
		minValue=list[0];
		for(T i=1;i<nModels;i++){
			if(list[i] > maxValue)maxValue=list[i];
			if(list[i] < minValue)minValue=list[i];
		}	
		bitArray=new BitArray(maxValue-minValue);
	 for(T i=0;i<nModels;i++){ 
	  bitArray->setBit(list[i]-minValue);
		} 
	} 
 BitIndex(PackedBitArray<T> *list,uint8_t nElementSize,size_t nModels){
		 maxValue=list->get(nElementSize,0);
		 minValue=maxValue;
		 for(uint32_t i=1;i<nModels;i++){
			const uint32_t value= list->get(nElementSize,i);
			if(value > maxValue)maxValue=value;
			if(value < minValue)minValue=value;
		}	
		bitArray=new BitArray(maxValue-minValue);
	 for(uint32_t i=0;i<nModels;i++){ 
	  bitArray->setBit(list->get(nElementSize,i)-minValue);
		} 
	}
	bool compare(PackedBitArray<T> *list,uint8_t modelBits, uint8_t nModels){
	 for(int i=0;i<nModels;i++){
	  size_t value= list->get(modelBits,i);
		 if(value < minValue || value > maxValue || ! getBit(value-minValue)){
		 	return(0);
		 }
		}
	}				 
	bool compare(T *list,T nModels){
	 for(int i=0;i<nModels;i++){
			if(list[i]<minValue || list[i] > maxValue || !getBit(list[i]-minValue)){
				//total_collisions++;
			 return(0);
			}
		}						
		return(1);
	} 
	unsigned char getBit(size_t index) {
  return (bitArray->getBit(index));
 }
 unsigned char setBit(size_t index) {
  return (bitArray->setBit(index));
 }
 ~BitIndex(){
  if(bitArray)delete bitArray; 
	}
};
template <class T>class DenseTrMatrix{
	//includes diagonal - single block of numbers - optimized for upper diagonal column major (FORTRAN style) access
	//not used for optimization but for storage requirements
 public:
 T *matrix;     //matrix
 int size; //number rows or columns 
 DenseTrMatrix(){
  matrix=0;
  size=0;
	}
 DenseTrMatrix(int m){
  matrix=new T [m*(m+1)/2];
  size=m;
	}		
	DenseTrMatrix(const DenseTrMatrix  &A){
	 size=A.size;
	 if(size){
	  matrix=new T [size*(size+1)/2];
   memmove(matrix,A.matrix,size*(size+1)/2*sizeof(T));
		}
		else{
			matrix=0;
		}	 
	}
 DenseTrMatrix & operator = (const DenseTrMatrix &rhs){	 
		if(rhs.size ==0){
		 if(size) delete [] matrix;
		 matrix=0;
		 size=0;
		}
		else if(rhs.size != size){
			if(size) delete [] matrix;
			size=rhs.size;
	  matrix=new T [size*(size+1)/2];
   memmove(matrix,rhs.matrix,size*(size+1)/2*sizeof(T));
		}
		else{
			memmove(matrix,rhs.matrix,size*(size+1)/2*sizeof(T));
		}
		return *this;	
	}
 void sq_to_tr(T *sqmatrix, int ldn)const{
		//leading dimension size provided - dense storage
		//upper column major expected
	 T *s=matrix;
	 size_t inc=sizeof(T);
	 size_t blocksize=inc;
	 for (int i=1;i<=size;i++){
			memmove(s,sqmatrix,blocksize);
			blocksize+=inc;
			s+=i;
			sqmatrix+=ldn;
		}	
	}
	void tr_to_sq (T *sqmatrix,int ldn)const{
		//leading dimension size provided - dense storage
		//upper column major expected
  //copies whol matrix
	 T *s=matrix;
	 size_t inc=sizeof(T);
	 size_t blocksize=sizeof(T);
	 for (int i=1;i<=size;i++){
			memmove(sqmatrix,s,blocksize);
			blocksize+=inc;
			s+=i;
			sqmatrix+=ldn;
		}	
	}
	void tr_to_sq_delj (T *d, int ldn,int j)const{
		//remove jth column
	 T *s=matrix;
	 size_t inc=sizeof(T);
	 size_t blocksize=inc;
	 int i=1;
	 while(i<=j){
			memmove(d,s,blocksize);
   blocksize+=inc;
			d+=ldn;
			s+=i;
			i++;
		}
		blocksize+=inc;
		s+=i;
		i++;	
	 while(i<=size){
			memmove(d,s,blocksize);
   blocksize+=inc;
			d+=ldn;
			s+=i;
			i++;
		}
	}
 void print()const{
	const int n=size*(size+1)/2;
	for(int row=0;row<size;row++){
  for(int col=0;col<row;col++){
			cout << 0 <<'\t';
		}
		int k=(row+1)*(row+2)/2-1;
		for(int col=row;col<size;col++){
		 cout << matrix[k] <<'\t';
		 k+=col+1;
		}
		cout <<endl; 
	}
	cout << endl;	
}	 
	~DenseTrMatrix(){
		if(matrix){
			delete[] matrix;
		}	
	}	
};
class DoubleHashTable{
	//uses hash of upper 16 and lower 16
	BitArray *filters[NFILTERS];
	public:
	DoubleHashTable(){
		for (int i=0;i<NFILTERS;i++)filters[i]=new BitArray(65536);
	}		
 
	bool countInsert(uint32_t hashValue,uint8_t nModels){
		uint16_t size=nModels %16;
		uint16_t uWord=hashValue >> 16 | size;
		uint16_t lWord=hashValue;
		return (filters[uWord%NFILTERS]->getSetBit(lWord));
	}
	void clear(){
  for (int i=0;i<NFILTERS;i++) filters[i]->clear();
	}
	~DoubleHashTable(){
	 for (int i=0;i<NFILTERS;i++)
	  delete filters[i];
	}
};
class ModelIndices{
	//unordered list of indices of models - can add and delete
	//save room in list for this - allocates maxModel space instead of nModel space
	public:
 	uint16_t maxModels=0;
  uint16_t nModels=0;
 	uint16_t *index=0;
 	uint16_t *list=0;
 	uint_fast32_t hashValue;
 	ModelIndices(): maxModels(0),nModels(0),index(0),list(0),hashValue(0){
		}
		ModelIndices(uint16_t h): maxModels(h),nModels(0),hashValue(0){
			index=new uint16_t[maxModels];
			memset(index,0,maxModels*sizeof(uint16_t));
			list=new uint16_t[maxModels];
		}
		ModelIndices(const ModelIndices &A):nModels(A.nModels),hashValue(A.hashValue){
			//doesn't copy any of the gunk that might be at the end of the list array
		 if(A.maxModels){
				if(maxModels != A.maxModels){
					maxModels=A.maxModels;
					if(list)delete[] list;
			  if(index)delete[] index;
			  index=new uint16_t[maxModels];
			  list=new uint16_t[maxModels];
				}
			 memmove(index,A.index,A.maxModels*sizeof(uint16_t));
			 memmove(list,A.list,A.maxModels*sizeof(uint16_t));
			}
			else{
				if(list)delete[] list;
			 if(index)delete[] index;
				list=0;
				index=0;
			}	
  }
  //full copy done when ModelIndices copied to modelIndices
   ModelIndices & operator = (const ModelIndices &rhs){
				nModels=rhs.nModels;
				hashValue=rhs.hashValue;
				if(rhs.maxModels){
				if(maxModels != rhs.maxModels){
					maxModels=rhs.maxModels;
					if(list)delete[] list;
			  if(index)delete[] index;
			  index=new uint16_t[maxModels];
			  list=new uint16_t[maxModels];
				}
			 memmove(index,rhs.index,rhs.maxModels*sizeof(uint16_t));
			 memmove(list,rhs.list,rhs.maxModels*sizeof(uint16_t));
			}
			else{
				if(list)delete[] list;
			 if(index)delete[] index;
				list=0;
				index=0;
			}
			return *this;
		}
		int insertElement(uint16_t m){ 
			if(index[m]) return(0); //no insert
			if(!index[m]){
			 list[nModels++]=m;  //add to end of list - we *want* nModels to be incremented before assignment
	   index[m]=nModels;	  //put a pointer to where the index is for deletion. This is incremented by 1 so that zero can indicate non-membership
			 hashValue= hashValue ^ hashLUT[m]; 
				return(1);
			}
			return 0;
		}
		uint_fast32_t insertElementHash(uint16_t m){
			if(index[m])return(0); //this element is already in set return 0;
   return(hashValue ^ hashLUT[m]);
		}
			uint16_t deleteElement_unordered(uint16_t m){
			//switches element with last element of list for quick removal
			//is not necessarily the same order as the variables in the R matrix
			//useful for hashing to see if set has been seen
			if(nModels <1){
				//fprintf(stderr,"trying to delete from an empty set\n");
				//exit(0);
				Rcpp::stop("trying to delete from an empty set\n");
			}
			if(nModels ==1){
				//empty set
				index[m]=0;
				nModels=0;
				hashValue=0;
				return(-1);
			}

			uint16_t i=index[m]-1; //index to be deleted
			if(i >=0){
				index[m]=0; //delete pointer
			 if(i != nModels-1){
					//put end of the list where the deletion is
			 	const uint16_t oldm=list[nModels-1];
			 	list[i]=oldm;
			 	index[oldm]=i+1;
			 }
			 	//adjust nModels and hash
				nModels--;
				hashValue=hashValue ^ hashLUT[m];
			 //return value of i for reordering
			}
			return(i);
		}

		uint_fast32_t deleteElement_unorderedHash(uint16_t m){
   return(hashValue ^ hashLUT[m]);  
		}
		int deleteElement(uint16_t m){
			//contracts list
			//keeps list in same order as variables in matrix
			if(nModels){
			 uint16_t i=index[m]-1; //index to be deleted
			 index[m]=0;
			 for(uint16_t k=i+1;k<nModels;k++){
			  list[k-1]=list[k];
			  index[list[k]]=k;
			 }	
			 nModels--;
    hashValue= hashValue ^ hashLUT[m] ; 
			}
			hashValue=0;
			return(0);
		}
		int order_deletion(uint16_t i){
			//converts an unordered deletion to one that matches R
			//need to know the ith column
			//check for case when the last column was actually delted
			if(i<0 || i >= nModels){
				return(0);
			}
			uint16_t temp=list[i];
			for(uint16_t k=i+1;k<nModels;k++){
			 list[k-1]=list[k];
			 index[list[k]]=k;
			}
			list[nModels-1]=temp;
			index[temp]=nModels; //add to index...
			return(1);
		}
		bool operator==(const ModelIndices &b)const{
	 if(nModels != b.nModels || hashValue != b.hashValue)return(0);
	 if(nModels==0)return(1);
  for (uint16_t i=0;i<nModels;i++){
			if(!b.index[list[i]])return(0);
		}
  return(1);	
	}

	void print_list()const{
		if(nModels == 0){
			Rcpp::Rcerr  << "NULL";
		}	
		for(uint16_t i=0;i<nModels;i++){
		 Rcpp::Rcerr  <<list[i] << '.'; 
		}
		Rcpp::Rcerr  << endl;
	}
	void print_ordered_list()const{
		if(nModels == 0){
			Rcpp::Rcerr  << "NULL";
		}
		uint16_t nSize=0;
		for(uint16_t i=0;i<maxModels && nSize < nModels;i++){
			if(index[i]){
				Rcpp::Rcerr  <<i << '.'; 
				nSize++;
			}	
		}	
		Rcpp::Rcerr  << endl;
	}
		~ModelIndices(){
			if(maxModels){
		  if(index)delete [] index;
		  if(list)delete [] list;
			}
		}			 
};
template <class T> class CompactModelIndices{
	public:
		uint_fast32_t hashValue;
	 uint8_t modelBits=gModelBits;
	 uint8_t nModels=0; //never more than 256 variables in a model
	 PackedBitArray <T>  *list=0;
	 CompactModelIndices(): hashValue(0){
	 	list=0;
	 	uint8_t modelBits=gModelBits; 
	 	uint8_t nModels=0;
	 }
	 CompactModelIndices(size_t _modelBits,size_t _nModels) : modelBits(_modelBits),nModels(_nModels),hashValue(0){
	 	list=new PackedBitArray<T>(modelBits,nModels);	
	 }
	 CompactModelIndices(const CompactModelIndices &A): hashValue(A.hashValue),modelBits(A.modelBits),nModels(A.nModels){
		 if(list){delete list;list=0;}
		 if(A.list){
				list=new PackedBitArray<T>(modelBits,nModels);
				const size_t tBits=sizeof(T)*8;
		  size_t nWords=((modelBits*nModels) & (tBits-1)) ? (modelBits*nModels)/tBits+1: (modelBits*nModels)/tBits;
				memmove(list->array,A.list->array,nWords*sizeof(T));
			}
	 }
	 CompactModelIndices & operator = (const CompactModelIndices &rhs){
			 hashValue=rhs.hashValue;
			 modelBits=rhs.modelBits;
			 nModels=rhs.nModels;
		 if(list){delete list;list=0;}
		 if(rhs.list){
				list=new PackedBitArray<T>(modelBits,nModels);
				const size_t tBits=sizeof(T)*8;
		  size_t nWords=((modelBits*nModels) & (tBits-1)) ? (modelBits*nModels)/tBits+1: (modelBits*nModels)/tBits;
				memmove(list->array,rhs.list->array,nWords*sizeof(T));
			}
			return *this;	
	 }
	 bool operator==(const CompactModelIndices &b)const{
			if(nModels == b.nModels &&  hashValue == b.hashValue){
			 return(1);
		  BitIndex<T> bitIndex(list,modelBits,nModels);
		  bool retvalue=bitIndex.compare(b.list,modelBits,nModels);
    return(retvalue);
			}	
	  return(0);
		}
	 CompactModelIndices( ModelIndices &mind): nModels(mind.nModels),hashValue(mind.hashValue){
		 if(list){delete list;list=0;}
		 if(nModels){
				const uint8_t modelBits=gModelBits;
		  list=new PackedBitArray<T>(modelBits,nModels);
    for(int i=0;i<mind.nModels;i++){
			  list->set(modelBits,i,mind.list[i]);
		  }
			}
		}
		CompactModelIndices & operator = (const ModelIndices &rhs){
	 	nModels=rhs.nModels;
	 	hashValue=rhs.hashValue;
		 if(list){delete list;list=0;}
		 if(rhs.nModels){
				const uint8_t modelBits=gModelBits;
		  list=new PackedBitArray<T>(modelBits,nModels);
    for(int i=0;i<rhs.nModels;i++){
			  list->set(modelBits,i,rhs.list[i]);
		  }
			}
			return *this;
		}
	 bool operator==(const ModelIndices &b)const{
			if(nModels == b.nModels &&  hashValue == b.hashValue){
			 return(1);
		  BitIndex<T> bitIndex(list,modelBits,nModels);
		  bool retvalue=bitIndex.compare(b.list,nModels);
    return(retvalue);
			}	
	  return(0);	
	 }
	 void print_list()const{
		 if(nModels == 0){
		 	Rcpp::Rcerr  << "NULL" << endl;
		 }	
		 for(int i=0;i<nModels;i++){
		  Rcpp::Rcerr  <<list->get(modelBits,i) << '.'; 
		 }
		 Rcpp::Rcerr  << endl;
	 }
	~CompactModelIndices(){
		 if(list)delete list;
	 }
};
namespace std {
		template <> 
		struct hash <CompactModelIndices<uint64_t>>{
   uint_fast32_t operator()(const CompactModelIndices<uint64_t>& k) const
   {
				const uint32_t nModels=k.nModels;
				const uint32_t size=nModels %16;	
				const uint8_t modelBits=k.modelBits;	
				if(!k.hashValue){
					uint_fast32_t hash=0;
		  //pairwise XOR each element
		   for (uint16_t i=0;i<k.nModels;i++){
      hash=hash ^ hashLUT[k.list->get(modelBits,i)];
		   }
		   return(hash & 0xFFFFFFF0 | size);
			 }
			 return(k.hashValue & 0xFFFFFFF0 | size);      
	 	}
  }; 
}

template <class T> class ModelIndicesHash{
	//class just using 32 or 64 bit hash representation of indices 
	public:
		T hashValue=0;
		uint8_t nModels=0;
		ModelIndicesHash():nModels(0),hashValue(0){}			
		ModelIndicesHash(uint8_t _nModels,T _hashValue):nModels(_nModels),hashValue(_hashValue){
		}	
		template <class T1> ModelIndicesHash (CompactModelIndices<T1> &SCMI){
	 	hashValue=SCMI.hashValue;
	 	nModels=SCMI.nModels;
	 }
	 ModelIndicesHash(ModelIndices& MI){
	 	hashValue=MI.hashValue;
	 	nModels=MI.nModels;
	 }
  ModelIndicesHash(const ModelIndicesHash &A): hashValue(A.hashValue),nModels(A.nModels){}
 	ModelIndicesHash & operator = (const ModelIndicesHash &rhs){
			hashValue=rhs.hashValue;
			nModels=rhs.nModels;
		}
	 bool operator==(const ModelIndicesHash &b)const{
			if( hashValue == b.hashValue && nModels == b.nModels){
			 return(1);
			}	
	  return(0);
		}
};
namespace std {
		template <> 
		struct hash <ModelIndicesHash<uint32_t>>{
   uint_fast32_t operator()(const ModelIndicesHash<uint32_t>& k) const
   {
				const uint32_t size=k.nModels %16;	
			 return(k.hashValue & 0xFFFFFFF0 | size);      
	 	}
  }; 
}

template <class T> class ModelSet{
	public:
	CompactModelIndices<uint64_t> modelIndices;
	T r2;
	T bic;
	double logprior;
 DenseTrMatrix <T> R;
 ModelSet() {
   modelIndices =0;
   r2 =0;
   bic =0;
   logprior=0;
   R=DenseTrMatrix<T>();
 }
 ModelSet(ModelIndices mInd, T mr2, T mbic, double mlogprior) {
   modelIndices=CompactModelIndices<uint64_t>(mInd);
   r2 = mr2;
   bic = mbic;
   logprior=mlogprior;
   R=DenseTrMatrix<T>();
 }
 ModelSet(CompactModelIndices<uint64_t> mInd, T mr2, T mbic,double mlogprior ) {
  modelIndices = mInd;
  r2 = mr2;
  bic = mbic;
  logprior=mlogprior;
  R=DenseTrMatrix<T>();
 }
 ModelSet( ModelIndices mInd, T mr2, T mbic,double mlogprior, T *sqR, int nRows,int ldn) {
   modelIndices = mInd;
   r2 = mr2;
   bic = mbic;
   R=DenseTrMatrix<T>(nRows);
   R.sq_to_tr (sqR,ldn);
   logprior=mlogprior;
 }

 ModelSet( CompactModelIndices<uint64_t> mInd, T mr2, T mbic,double mlogprior,T *sqR, int nRows,int ldn) {
  modelIndices = mInd;
  r2 = mr2;
  bic = mbic;   
  R=DenseTrMatrix<T>(nRows);
  R.sq_to_tr (sqR,ldn);
  logprior=mlogprior;
 }
	ModelSet (const ModelSet &A){
	 R=A.R;
	 modelIndices=A.modelIndices;				
  bic=A.bic;
		r2=A.r2;   
	 logprior=A.logprior; 
 }
 ModelSet & operator = (const ModelSet &rhs){
				//deep copy of R
  R=rhs.R;
		bic=rhs.bic;
		r2=rhs.r2; 
		modelIndices=rhs.modelIndices;
		logprior=rhs.logprior; 	
	}
	//bool operator<(const ModelSet& b) const { return (bic < b.bic) && ! (modelIndices == b.modelIndices); }
 //bool operator>(const ModelSet& b) const { return (bic > b.bic) && ! (modelIndices == b.modelIndices); }
 bool operator<(const ModelSet& b) const { return (bic < b.bic) && ! (modelIndices == b.modelIndices); }
 bool operator>(const ModelSet& b) const { return (bic > b.bic) && ! (modelIndices == b.modelIndices); }
 bool operator>=(const ModelSet& b) const { return !(*this < b); }
 bool operator<=(const ModelSet& b) const { return !(*this > b); }
	bool operator==(const ModelSet &b)const{
		return(modelIndices == b.modelIndices);
	}
	bool operator!=(const ModelSet& b) const { return !(*this == b); }
};

//for sorting by edgeweights we need data structure where there is a pair and a score
//simple edge list class with weights and sort routine


class EdgeList{
	public:
 int **parents;
 float **edgeWeights;
 int *nParents;
 int nNodes;
 EdgeList(){
		nNodes=0;parents=0;nParents=0;edgeWeights=0;
	}
	EdgeList(int nGenes,float minWeight,float **weights){ 
		nNodes=nGenes;
		nParents=new int[nGenes];
		memset(nParents,0,sizeof(int)*nGenes);
		//get sizes
		for(int i=0;i<nGenes;i++){ //children
			for(int j=0;j<nGenes;j++){ //parents;
			 if(weights[i][j] > minWeight){
     nParents[i]++;
			 }	
			}
		}
		//allocate
	 edgeWeights=new float*[nGenes];
	 parents=new int*[nGenes];
		for(int i=0;i<nGenes;i++){
			edgeWeights[i]=0;
			parents[i]=0;
			if(nParents[i]){
			 edgeWeights[i]=new float [nParents[i]];
			 parents[i]=new int [nParents[i]];
			}
		}
		memset(nParents,0,sizeof(int)*nGenes);
		for(int i=0;i<nGenes;i++){ //children
			for(int j=0;j<nGenes;j++){ //parents
			 if(weights[i][j] > minWeight){
			 	parents[i][nParents[i]]=j;
			 	edgeWeights[i][nParents[i]]=weights[i][j];
     nParents[i]++;
			 }
			}	
		}		
	}
			
 EdgeList(int n){
		nNodes=n;
		parents=new int*[n];
		nParents=new int[n];
		edgeWeights=new float*[n];
		for (int i=0;i<n;i++){
			parents[i]=0;
			nParents[i]=0;
			edgeWeights[i]=0;
		}	
	}
	EdgeList(int _nNodes,set<pair<pair<int,int>,float>> edgeSet){
		nNodes=_nNodes;
		parents=new int*[nNodes];
		nParents=new int[nNodes];
		edgeWeights=new float*[nNodes];
		for (int i=0;i<nNodes;i++){
			parents[i]=0;
			nParents[i]=0;
			edgeWeights[i]=0;
		}
		//find sizes of parents

		for(auto f : edgeSet){
			nParents[f.first.second]++;
		}
		//allocate
		for (int i=0;i<nNodes;i++){
			if(nParents[i]){
			 parents[i]=new int[nParents[i]];
			 edgeWeights[i]=new float[nParents[i]];
			 nParents[i]=0; //so that it can be used as a counter
			}
		}
		//assign values
		for(auto f : edgeSet){
			const int c=f.first.second;
			parents[c][nParents[c]]=f.first.first;
			edgeWeights[c][nParents[c]]=f.second;
			nParents[c]++;
		}			
	}			
	EdgeList(int _nNodes,vector<int>nEdges,vector<int>inParents,vector<int>inChildren,vector<double>inWeights){
		nNodes=_nNodes;
		parents=new int*[nNodes];
		nParents=new int[nNodes];
		edgeWeights=new float*[nNodes];
		for (int i=0;i<nNodes;i++){
			parents[i]=0;
			nParents[i]=0;
			edgeWeights[i]=0;
		}
		int totalEdges=0;
		for(unsigned int i=0;i<inChildren.size();i++){
			const int c=inChildren[i];	
			nParents[c]=nEdges[i];
			parents[c]=new	int[nEdges[i]];
			memmove(parents[c],&(inParents[totalEdges]),nEdges[i]*sizeof(int));
			edgeWeights[c]=new	float[nEdges[i]];
			double *weights=&inWeights[totalEdges];
		 for(int k=0;k<nEdges[i];k++) edgeWeights[c][k]=weights[k];
		 totalEdges+=nEdges[i];
		}	
	}	
		
	EdgeList(const EdgeList &A){
		nNodes=A.nNodes;
		parents=new int*[nNodes];
		nParents=new int[nNodes];
		edgeWeights=new float*[nNodes];
		memmove(nParents,A.nParents,nNodes*sizeof(int));
		for (int i=0;i<nNodes;i++){
			if(nParents[i]){
				parents[i]=new int[nParents[i]*sizeof(int)];
				memmove(parents[i],A.parents[i],nParents[i]*sizeof(int));
				edgeWeights[i]=new float[nParents[i]*sizeof(float)];
				memmove(edgeWeights[i],A.edgeWeights[i],nParents[i]*sizeof(float));	 
			}	
			else{
			 parents[i]=0;
			 nParents[i]=0;
			 edgeWeights[i]=0;
			}
		}	
	}

 ~EdgeList(){
		for (int i=0;i<nNodes;i++){
			if(nParents[i]){
				delete[] parents[i];
				delete[] edgeWeights[i];
			}	
		}
		if(nNodes){
			delete[] nParents;delete[] edgeWeights;delete[] parents;
		}		
	}
  EdgeList nonSelfList(){
		EdgeList nonSelfList(nNodes);
  //count edges
 	for(int i=0;i<nNodes;i++){
			int j;
			for (j=0;j<nParents[i];j++){
			 if(parents[i][j] == i && edgeWeights[i][j] > 0){
					//allocate parents 
					//copy every thing but i
					nonSelfList.nParents[i]=nParents[i]-1;
					if(nonSelfList.nParents[i]){
						nonSelfList.parents[i]=new int[nonSelfList.nParents[i]*sizeof(int)];
						nonSelfList.edgeWeights[i]=new float[nonSelfList.nParents[i]*sizeof(float)];
						if(j){
							memmove(nonSelfList.parents[i],parents[i],j*sizeof(int));
							memmove(nonSelfList.edgeWeights[i],edgeWeights[i],j*sizeof(float));		
						}	
						if(j<nonSelfList.nParents[i]){
							memmove(nonSelfList.parents[i]+j,parents[i]+j+1,(nonSelfList.nParents[i]-j)*sizeof(int));
							memmove(nonSelfList.edgeWeights[i]+j,edgeWeights[i]+j+1,(nonSelfList.nParents[i]-j)*sizeof(float));		
						}	
					}	
					break;
				}	
			}
			//check if loop finished - if it did copy the old parents over
			if(j==nParents[i]){
				nonSelfList.nParents[i]=nParents[i];
				if(nParents[i]){
					nonSelfList.parents[i]=new int[nParents[i]*sizeof(int)];
				 memmove(nonSelfList.parents[i],parents[i],nParents[i]*sizeof(int));
				 nonSelfList.edgeWeights[i]=new float[nParents[i]*sizeof(float)];
				 memmove(nonSelfList.edgeWeights[i],edgeWeights[i],nParents[i]*sizeof(float));		
				}	
			}	
		}
		return(nonSelfList);
	}	
	void printEdges(float edgeMin,vector<string> headings,bool selfie,bool keepNegative,float pruneEdgeMin){
		for(int i=0;i<nNodes;i++){
		 for (int j=0;j<nParents[i];j++){
				if(edgeWeights[i][j] < 0 && edgeWeights[i][j] > -pruneEdgeMin){
					edgeWeights[i][j]=-edgeWeights[i][j];
				}	
			 if((selfie || parents[i][j] != i) && (edgeWeights[i][j] >= edgeMin || (keepNegative && -edgeWeights[i][j] >= edgeMin))){
					if(headings.size()){
			   cout << headings[parents[i][j]] << '\t' << headings[i]<< '\t' << edgeWeights[i][j] <<endl;
					}
					else{
					cout << parents[i][j] << '\t' << i << '\t' << edgeWeights[i][j] <<endl;	
					}	
			 }
			} 
		}	
	}
	void printEdges(float edgeMin,vector <string>headings,float *adjMatrix,bool selfie,float pruneEdgeMin){
		for(int i=0;i<nNodes;i++){
			float* const slice=adjMatrix+i*nNodes;
		 for (int j=0;j<nParents[i];j++){
				if(edgeWeights[i][j] < 0 && edgeWeights[i][j] > -pruneEdgeMin){
				 edgeWeights[i][j]=-edgeWeights[i][j];
			 }	
			 if((selfie || parents[i][j] != i) && edgeWeights[i][j] >= edgeMin && slice[j] < 2.0f){
					if(headings.size()){
			   cout << headings[parents[i][j]] << '\t' << headings[i]<< '\t' << edgeWeights[i][j] <<endl;
					}
					else{
					cout << parents[i][j] << '\t' << i << '\t' << edgeWeights[i][j] <<endl;	
					}	
			 }
			} 
		}	
	}
	void printSelfEdges(float edgeMin,vector<string> headings,bool keepNegative,float pruneEdgeMin){
		for(int i=0;i<nNodes;i++){
		 for (int j=0;j<nParents[i];j++){
				if(edgeWeights[i][j] < 0 && edgeWeights[i][j] > -pruneEdgeMin){
					edgeWeights[i][j]=-edgeWeights[i][j];
				}	
			 if(parents[i][j] == i && (edgeWeights[i][j] >= edgeMin || (keepNegative && -edgeWeights[i][j] >= edgeMin))){
					if(headings.size()){
			   cout << headings[parents[i][j]] << '\t' << headings[i]<< '\t' << edgeWeights[i][j] <<endl;
					}
					else{
					cout << parents[i][j] << '\t' << i << '\t' << edgeWeights[i][j] <<endl;	
					}	
			 }
			} 
		}	
	}
	float* convert_to_pvalue_matrix(){
  float *adjMatrix=new float[nNodes*nNodes];
  for(int i=0;i<nNodes*nNodes;i++){
			adjMatrix[i]=0; //indicates noEdge
		}	
		for(int i=0;i<nNodes;i++){
		 for (int j=0;j<nParents[i];j++){
			 adjMatrix[parents[i][j]*nNodes+i]=edgeWeights[i][j];
			}
		}
		return(adjMatrix);			
	}
	float** convert_to_logodds(){
		float **logWeights=new float* [nNodes];		
	 for(int i=0;i<nNodes;i++){
			if(nParents[i]){
				logWeights[i]=new float[nParents[i]];
			}
			else logWeights[i]=0;	
		 for (int j=0;j<nParents[i];j++){
				if(edgeWeights[i][j]){
		 	 logWeights[i][j]=-log(edgeWeights[i][j]);
		 	 logWeights[i][j]=(logWeights[i][j] <0)? 0 : logWeights[i][j];
				} 
		 	else
		 	 logWeights[i][j]=0; 
   }
	 }
	 return(logWeights);
	}
	
float dijkstra_limit(const int source,const int destination,const float limit, float **logWeights, float tol){
 float *d=new float[nNodes];
 int path_found=0;
 for(int i = 0 ;i < nNodes; i++){
  d[i] = std::numeric_limits<float>::max();
 }
 priority_queue<pair<int,float>, vector<pair<int,float> >, Comparator> Q;
 d[destination] = 0.0f; //semantics - we start at the destination and work backwards to the source
 Q.push(make_pair(destination,d[destination]));
 while(!Q.empty()){
  int u = Q.top().first;
  if(u==source){ 
			path_found=1;
			break; //found shortest path
		}
  Q.pop();

  for(int i=0; i < nParents[u]; i++){
			if(logWeights[u][i] >=0){ //less than 0 indicates edge has been deleted
    const int v= parents[u][i];
    if ((u!=destination || v != source) && u != v){
     const float w = logWeights[u][i];
     if(d[v] > d[u]+w){
      d[v] = d[u]+w;
      if(d[v] <= limit +tol ){ //don't push parents onto queue unless the path is still <= the confidence of "direct" path and the edge not the direct path
       Q.push(make_pair(v,d[v]));
		  		}
     }
			 }
			} 
  }
 }
 if (!Q.empty()){
		return(d[Q.top().first]);
	}
	else{ 
	 return (-1);
	}	
 delete[] d;
}

 pair<int,int>* sort_by_edge_weights(float **logWeights,int *nEdges,float edgeMin){
	 //returns a sorted array of pairs node - parent_index
	 //returns the logWeights
	 int my_nEdges=0;	
	 for(int i=0;i<nNodes;i++){
	 	for (int j=0;j<nParents[i];j++){
	  	if(edgeWeights[i][j] > edgeMin) my_nEdges++;
			}
		}
	 pair <int,int> *nodeParent=new pair<int,int>[my_nEdges];	 
	 pair <int,int> *temp=new pair<int,int>[my_nEdges];
	 int *sortIndex=new int[my_nEdges];
	 float *scores=new float[my_nEdges];
	 int n=0;
  for (int i=0;i<nNodes;i++){
		 for (int j=0;j<nParents[i];j++){
		 	if(edgeWeights[i][j] > edgeMin){
		 	 temp[n].first=i;
		 	 temp[n].second=j;
		 	 scores[n++]=logWeights[i][j];
			 }
		 }
	 }
	 sort_by_scores(my_nEdges,scores,sortIndex,0);
	 for	(int i=0;i<my_nEdges;i++){
			nodeParent[i]=temp[sortIndex[i]];
		}
		delete[] temp;
		delete[] scores;
		*nEdges=my_nEdges;
		return(nodeParent);		
 }

	int prune_edges(float edgeMin,float edgeTol){
		float **logWeights=convert_to_logodds();

		int nEdges=0;
		pair<int,int> *nodeParent= sort_by_edge_weights(logWeights,&nEdges,edgeMin); 
		
		for(int i=0;i<nEdges;i++){
		 const int endNode=nodeParent[i].first;
		 const int j=nodeParent[i].second;
		 const float limit=logWeights[endNode][j];			 
		 float tol=(edgeTol >=0) ? edgeTol : 0;
			const float retvalue=dijkstra_limit(parents[endNode][j],endNode,limit,logWeights,tol);
			if(retvalue != -1){
			 //just set the i j of logWeights <0 to mark this - simpler than updating all the sizes
			 logWeights[endNode][j]=-1;
			}

		}
		for(int i=0;i<nEdges;i++){
			const int endNode=nodeParent[i].first;
		 const int j=nodeParent[i].second;
		 if(logWeights[endNode][j]<0)edgeWeights[endNode][j]=-edgeWeights[endNode][j];
		}
		return(1);
	}
	float* cutter_prune_edges(float lower_threshold,float upper_threshold){
		//negative numbers are indicated for deletion in multithread
	 float *adjMatrix=convert_to_pvalue_matrix();
	 for (int k = 0; k < nNodes; k++){
			int m=0;
			float* const slicek=adjMatrix+k*nNodes;
		 for (int i = 0; i <nNodes; i++){
				float* const slicei=adjMatrix+i*nNodes;
			 for (int j = 0; j < nNodes; j++){
				 const float edge = 1.0f-adjMatrix[m];
				 if (edge<lower_threshold){
					 float viaK =  fmaxf(fabsf(slicei[k]), fabsf(slicek[j]) );
					 if ( fabsf(edge) > viaK ){
						 slicei[j] = -viaK;
					 }
					}
					m++;
				}
			}
		}
	 for (int i = 0; i < nNodes*nNodes; i++){
		 float val = adjMatrix[i];
		 if (((*((unsigned int*)&val) & 2147483648u))   or   (val >= upper_threshold)  )adjMatrix[i] = 2.0f;
	 }
	 return(adjMatrix);
	}
};	
//class dependent globals

//main subroutines
template <class T> int findRegulators(T g,int optimizeBits,int maxOptimizeCycles,float uPrior,float twoLogOR,int nVars,int nThreads,bool rankOnly,int geneIndex,T **data,T **rProbs,int *parents, double *postProbs, T *A, T *ATA, int Aldr, int ATAldr,int nGenes,int nRows,int nSamples,int nTimes);
//template <class T> T** readPriorsMatrix(string priorsFile,int &genes);
//template <class T> void readPriorsList(string priorsListFile,vector <string> names, T **priors, T uniform_prob);
//EdgeList* readEdgeListFile (string edgeListFile,vector<string> &names);
//template <class T> T** readTimeData(string matrixFile,vector <string> &headers,int &nGenes,int &nSamples,int &nRows,int &nTimes,bool noHeader,bool useResiduals,string residualsFile);
//template <class T> T** readData(string matrixFile,vector <string> &headers,int &nGenes,int &nSamples,bool noHeader);
template <class T> void initRegressParms(T *A, T *ATA, T **data,int nGenes,int nRows, int nSamples,int nTimes,int &nVars, int nThreads,bool timeSeries);

//scanBMA with g prior
template <class T> int fastScanBMA_g(T *mATA, T *mATb,T btb,T sst, int ignoreIndex, T *priorProbs,bool rankOnly,double *postProbs,int *parents, int nRows, int nCols, int nVars, double twoLogOR ,double g, int optimizeBits,int maxOptimizeCycles);

//routine for choosing best sets of models - repeated multiple times to optimize for parameters
template <class T> T chooseBestModels(double g,T *ATA,int nVars,int nRows,int nCols,int *pord,T *ATb,T sst, T btb, double *postProbs, int *parents, int *npostProbs, double *logpriors,double twoLogOR);

//calculate the sst
template <class T> T calculate_sst(int sizeb, T *b,T *ones);

//needed for copying - overloading = requires both classes to be defined - no easy way to predefine classes
template <class T> void copy_indices(ModelIndices &dest,const CompactModelIndices<T>  &source);

//cholesky routines to calculate correlation coeff
template <class T> T getR2_full(ModelIndices &modelIndices,T* ATA,int ATAldr, T* ATb,const T btb,T *R,int Rldr);
template <class T> T getR2_down(int nRows,ModelIndices &modelIndices,T* ATb,const T btb,T *R, int Rldr, int dCol);
template <class T> T getR2_up(ModelIndices &modelIndices,T* ATA,int ATAldr, T* ATb,const T btb,T *R,int Rldr);

//givens rotation on columns for Cholesky downdate
template <class T> void qhqr (int nRows,int nCols,T *R, int ldr,T *c,T *s);

//LAPACK FORTRAN subroutines needed for givens - drotg is too inaccurate - 
extern "C"{
 extern void dlartg_ ( double* f, double* g, double* cs, double* sn, double* r );
 extern void slartg_ ( float* f, float* g, float* cs, float* sn, float* r );
}

//wrappers for OpenBLAS/LAPACK - can be easily modified to use regular BLAS
//right now the inputs are simplified for the fastBMA use - the full set of input parms can be added later for full BLAS function if needed

//multiply vector by transpose of matrix	wraps dgemv/sgemv
void mtrv(int nRows, int nCols, float *A, int Aldr, float *b, float *ATb);
void mtrv(int nRows, int nCols, double *A, int Aldr, double *b, double *ATb);

//square the matrix AT*A  wraps dgemm/sgemm
void sqmm(int nRows,int nCols,float *A,int Aldr, float *ATA,int ATAldr);
void sqmm(int nRows,int nCols,double *A,int Aldr, double *ATA,int ATAldr);
void sqmm(int nRows,int nCols,float *A,int Aldr, float *ATA,int ATAldr,int nThreads);
void sqmm(int nRows,int nCols,double *A,int Aldr, double *ATA,int ATAldr,int nThreads);
//lartg LAPACK's version of rotg wraps dlartg/slartg
void lartg (double* f, double* g, double* cs, double* sn, double* r);
void lartg (float* f, float* g, float* cs, float* sn, float* r);

//dot product - wraps ddot/sdot
double dot(int n,double *x,double *y);
float dot(int n,float *x,float *y);

//vector addition wraps daxpy/saxpy
void axpy (int n,double a,double *x,double *y);
void axpy (int n,float a,float *x,float *y);

//triangle solver wraps dtrsv/strsv
void trsvutr(int n,double *R, int Rldr, double *v);
void trsvutr(int n,float *R, int Rldr, float *v);

//cholesky wraps dpotrf/spotrf
void potrf(char ul ,int n,float *R,int Rldr);
void potrf(char ul ,int n,double *R,int Rldr);

//functor for optimization - allows for just the parameter to be optimized to be exposed to the optimization library routines
template <class T> class BMAoptimizeFunction{
  public:
	 BMAoptimizeFunction  (T *ATA,int nVars,int nRows,int nCols,int *pord,T *ATb,T sst, T btb, double *postProbs, int *parents, int *npostProbs, double *logpriors,double twoLogOR) : _ATA(ATA), _nVars(nVars), _nRows(nRows), _nCols(nCols), _pord(pord), _ATb (ATb), _sst (sst), _btb(btb), _postProbs(postProbs), _parents(parents), _npostProbs (npostProbs), _logpriors(logpriors), _twoLogOR(twoLogOR){}
	 double operator () (double g0){
			return(chooseBestModels <T> (g0,_ATA,_nVars,_nRows,_nCols,_pord,_ATb, _sst, _btb,_postProbs, _parents, _npostProbs, _logpriors,_twoLogOR));
		} 
	private:
	T *_ATA;
	int _nVars;
	int _nRows;
	int _nCols;
	int *_pord;
	T *_ATb;
	T _sst;
	T _btb;
	double *_postProbs;
	int *_parents;
	int *_npostProbs;
	double *_logpriors;
	double _twoLogOR;
};
template <class T> int fastScanBMA_g(T *mATA, T *mATb,T btb,T sst, int ignoreIndex, T *priorProbs,bool rankOnly,double *postProbs,int *parents, int nRows, int nCols, int nVars, double twoLogOR ,double g, int optimizeBits,int maxOptimizeCycles){
 double *logpriors=new double[nVars]; //always keep this as double as the differences may be very small
 int npostProbs=0; //counter for final number of postProbs; 
 //sort indices decreasing by scores
 //check for underflow
 int *pord=new int[nCols];
 sort_by_scores(nCols,priorProbs,pord,0);
 if(rankOnly){
		T logj=log(UNIFORM_PRIOR)-log(1.0-UNIFORM_PRIOR);
		if(ignoreIndex >= 0){
		int j=0;
		int i=0;
		while(j <nVars && i <nCols){
			if(pord[i] != ignoreIndex){
				logpriors[j]=logj;
		  pord[j++]=pord[i];
		 }
		 i++;
		}
	 nVars=j;
	}	
 else{ 
  for (int i=0;i<nVars;i++)
   logpriors[i]=logj;
	 }	
	}
	else{	
  if(ignoreIndex >= 0){
		 int j=0;
		 int i=0;
		 while(j <nVars && i <nCols){
			 if(pord[i] != ignoreIndex){
				 logpriors[j]=log(priorProbs[pord[i]])-log(1.0-priorProbs[pord[i]]);
		   pord[j++]=pord[i];
		 	}
		  i++;
		 }
	 	nVars=j;
	 }	
  else{ 
   for (int i=0;i<nVars;i++){
    logpriors[i]=log(priorProbs[pord[i]])-log(1.0-priorProbs[pord[i]]);
	  }
		} 
	}

 //here we will use the sorted indexing from 1 to nVar - nVars may be smaller than nCols
  T *ATA= new T [(nVars+1)*(nVars+1)];
  T *ATb= new T [nVars+1];

  //reorder mATA and mATb to match sorted ordering
  //could do indirect addressing but copying is probably scales better especially when there are multiple threads  
  {
   //first column/row reserved ones "variable"
   //so if i/j is zero map to zero otherwise map to pord[i]/pord[j]
   //copy 1s column
   
   //first row is ones variable x ones variable
   ATA[0]=mATA[0];
   ATb[0]=mATb[0];
   
   //next rows are from the real variables
   for(int rowATA=1;rowATA<nVars+1;rowATA++){
		  ATA[rowATA]=mATA[pord[rowATA-1]+1];
			}
			
			//copy other columns - careful with row corresponding to ones variable 
   for(int colATA=1;colATA<nVars+1;colATA++){
	  	const int colmATA=pord[colATA-1]+1;
    ATb[colATA]=mATb[colmATA];
    T* const mATAslice=mATA+(colmATA*(nCols+1));
    T* const ATAslice=ATA+(colATA*(nVars+1));
    ATAslice[0]=mATAslice[0];
    for(int rowATA=1;rowATA<nVars+1;rowATA++){
					ATAslice[rowATA]=mATAslice[pord[rowATA-1]+1];
		 	}
			}	
	 }
	
	int nIterations=0;
	bool doneOptimizing=0;
	if(optimizeBits){
		//do a minimization for g
		typedef std::pair<T, T> Result;
		double g0=g; 
		boost::uintmax_t max_iter=maxOptimizeCycles;
		Result r2 = boost::math::tools::brent_find_minima(BMAoptimizeFunction<T>(ATA,nVars,nRows,nCols,pord,ATb,sst,btb,postProbs,parents,&npostProbs,logpriors,twoLogOR), (T) 1, (T)nRows, optimizeBits, max_iter);
  //std::cout << "g=" << r2.first << " f=" << r2.second << std::endl;
	}
	else{
		//just use the current g		

		chooseBestModels(g,ATA,nVars,nRows,nCols,pord,ATb,sst,btb,postProbs,parents,&npostProbs,logpriors,twoLogOR);
	}
 delete [] ATA;
 delete [] ATb;
 delete [] logpriors;
 delete [] pord;
 return(npostProbs);
}

template <class T> T chooseBestModels(double g,T *ATA,int nVars,int nRows,int nCols,int *pord,T *ATb,T sst, T btb, double *postProbs, int *parents, int *npostProbs, double *logpriors,double twoLogOR){
  //start loop here when optimizing g0
  DoubleHashTable gHashTable;
  std::set<ModelSet <T> > keepModels;
  std::set<ModelSet <T> > activeModels;
  std::set<ModelSet <T> > nextModels; 
  ModelIndices modelIndices(nVars);
  activeModels.insert(ModelSet<T>(modelIndices, 0, 0, 0));
  T minBIC = 0;
  T candidateR2 = 0;
  T candidateBIC = 0;
  T cutoffBIC = twoLogOR;

  typename std::set<ModelSet <T> >::iterator it,it1;
  std::unordered_set<ModelIndicesHash<uint32_t>> checkedModels;
 // Loop through while we have active models
 // to search around
  int curpass=0;
  while ( (int)activeModels.size() > 0) {
   curpass++;
   for ( it = activeModels.begin(); it != activeModels.end(); it++ ){
 			copy_indices(modelIndices,it->modelIndices);
 			ModelIndices tempmodelIndices=modelIndices;
 		 //Do a first pass to see which inserts/deletes are needed
 		 //This saves overhead of creating and copying data to large arrays
 		 
 			const double current_logprior=it->logprior;
 			int inserts[nVars];
 			int deletes[nVars];
 			int ndeleted=0;
 			int ninserted=0;
 			//see which changes need to be made
 		 for (int i = 0; i < nVars; i++ ) {
					const uint8_t nInsertModels=modelIndices.nModels+1;
					const uint8_t nDeleteModels=modelIndices.nModels-1;
 			//go through active models and add or remove model i
 			 //modelIndices.print_list();
     if (modelIndices.nModels && modelIndices.index[i]){
 				 if (modelIndices.nModels >1){
       if(!gHashTable.countInsert(tempmodelIndices.deleteElement_unorderedHash(i),nDeleteModels)){
 						 deletes[ndeleted++]=i;
 						}
 						tempmodelIndices=modelIndices;
 					}		
 				}
     else{
      if ( !gHashTable.countInsert(tempmodelIndices.insertElementHash(i),nInsertModels)){							
 						inserts[ninserted++]=i;
 					}
 					tempmodelIndices=modelIndices;	
 				}
 			}
 			//now pass all the non-duplicated sets and determine the rsquared coefficients 
 			//do inserts/denovo
 			const int nModels=modelIndices.nModels;
 			if(ninserted){
 				const int p=nModels+2;
 				//de novo
 				if(nModels <1){
 					T *Rarray=new T[(p+1)*(p+1)];
 					for(int k=0;k<ninserted;k++){
 						//add model i
 						const int i=inserts[k];
 						const double logprior=current_logprior+logpriors[i];
 						tempmodelIndices=modelIndices;	
 						tempmodelIndices.insertElement(i);
 					 T rv=getR2_full(tempmodelIndices,ATA,nVars+1,ATb,btb,Rarray,p+1);
 						candidateR2 = 1.0-(rv/sst);
 						candidateBIC = (nRows-1) * log(1 + g*(1-candidateR2)) + (1 + (int)tempmodelIndices.nModels- nRows) * log(1 + g)-2.0*logprior;
     		if ( candidateBIC - minBIC < twoLogOR ) {   
							 nextModels.insert(ModelSet<T>(tempmodelIndices, candidateR2, candidateBIC,logprior,Rarray,p,p+1));
							 minBIC = ((minBIC < candidateBIC) ? minBIC : candidateBIC);;
							}
					 } 
					 delete [] Rarray;
				 }
				 //insert	
				 else{
						T *Rarray=new T[(p+1)*(p+1)];
				  T *Rplus=new T [p*p];
				  it->R.tr_to_sq(Rplus,p);
			 	 for(int k=0;k<ninserted;k++){
							bool denovo=0;
						 const int i=inserts[k];
						 const double logprior=current_logprior+logpriors[i];
						 tempmodelIndices=modelIndices;
			 	  tempmodelIndices.insertElement(i);
						 T rv=getR2_up(tempmodelIndices,ATA,nVars+1,ATb,btb,Rplus,p);
						 //check if rv is reasonable - otherwise use the full method
						 if(boost::math::isfinite(rv) || rv <0 || rv > sst){
						  rv=getR2_full(tempmodelIndices,ATA,nVars+1,ATb,btb,Rarray,p+1);
						  denovo=1;
							} 
						 candidateR2 = 1.0-(rv/sst);
						 candidateBIC = (nRows-1) * log(1 + g*(1-candidateR2)) + (1 + (int)tempmodelIndices.nModels- nRows) * log(1 + g)-2.0*logprior;
						 if ( candidateBIC - minBIC < twoLogOR ) { 
							 if(denovo){
									nextModels.insert(ModelSet<T>(tempmodelIndices, candidateR2, candidateBIC,logprior,Rarray,p,p+1));
								}
								else{
									nextModels.insert(ModelSet<T>(tempmodelIndices, candidateR2, candidateBIC,logprior,Rplus,p,p));
								}
						 	minBIC = ((minBIC < candidateBIC) ? minBIC : candidateBIC);								
					 	}
				 	}
			 		delete[] Rplus;
			 		delete[] Rarray;
			 	}	
			 }	
			 //check deletes
			 if(ndeleted){
				 const int p=nModels;
				 //de novo
			  if(p <=2){
					 T *Rarray=new T[(p+1)*(p+1)];
      for(int k=0;k<ndeleted;k++){
						 const int i=deletes[k];
						 const double logprior=current_logprior-logpriors[i];
					  tempmodelIndices=modelIndices;	
						 int delete_index=tempmodelIndices.deleteElement_unordered(i);
					  T rv=getR2_full(tempmodelIndices,ATA,nVars+1,ATb,btb,Rarray,p+1);
       candidateR2 = 1.0-(rv/sst);
       candidateBIC = (nRows-1) * log(1 + g*(1-candidateR2)) + (1 + (int)tempmodelIndices.nModels- nRows) * log(1 + g)-2.0*logprior;
       if ( candidateBIC - minBIC < twoLogOR ) {
								tempmodelIndices.order_deletion(delete_index);
        nextModels.insert(ModelSet<T>(tempmodelIndices, candidateR2, candidateBIC,logprior,Rarray,p,p+1));
        minBIC = ((minBIC < candidateBIC) ? minBIC : candidateBIC);	    
       }
				 	} 
				 	delete [] Rarray;
				 }
				 //delete	
				 else{
						T *Rarray=new T[(p+1)*(p+1)];
					 T *Rminus=new T[(p+1)*(p+1)];
	     for(int k=0;k<ndeleted;k++){
							bool denovo=0;
						 const int i=deletes[k];
						 const double logprior=current_logprior-logpriors[i];
						 tempmodelIndices=modelIndices;
				 	 int delete_index=tempmodelIndices.deleteElement_unordered(i);
				 	 //the first column (0) is the column of ones but the index stored is 1 greater than the modelIndex 
				 	 int j=modelIndices.index[i]; 
       //copy matrix and delete jth colum
       it->R.tr_to_sq_delj(Rminus,p+1,j);
						 T rv=getR2_down(p+1,modelIndices,ATb,btb,Rminus,p+1,j);
						 if(boost::math::isfinite(rv) || rv <0 || rv > sst){
			     rv=getR2_full(tempmodelIndices,ATA,nVars+1,ATb,btb,Rarray,p+1);
						  denovo=1;
							} 
       candidateR2 = 1.0-(rv/sst);
       candidateBIC = (nRows-1) * log(1 + g*(1-candidateR2)) + (1 + (int)tempmodelIndices.nModels- nRows) * log(1 + g)-2.0*logprior;
       if ( candidateBIC - minBIC < twoLogOR ) {
					 		tempmodelIndices.order_deletion(delete_index);
					 		if(denovo) nextModels.insert(ModelSet<T>(tempmodelIndices, candidateR2, candidateBIC,logprior,Rarray,p,p+1));
        else nextModels.insert(ModelSet<T>(tempmodelIndices, candidateR2, candidateBIC,logprior,Rminus,p,p+1));								
        minBIC = ((minBIC < candidateBIC) ? minBIC : candidateBIC);
       }
				 	}
				 	delete[] Rminus;
				 	delete[] Rarray;
			 	}
    }	
		 }
   cutoffBIC = minBIC + twoLogOR;

   // Remove models with bic greater than the cutoff

   uint64_t modelCount=0;
   it = keepModels.begin();
   while( it != keepModels.end() && (*it).bic <= cutoffBIC  && modelCount< gMaxKeptModels) {
    it++;
    modelCount++;
   }
   keepModels.erase(it, keepModels.end());
   if(keepModels.size()==gMaxKeptModels){
				//decrease OR
				cutoffBIC=it->bic;
				twoLogOR=cutoffBIC-minBIC;
			}	 

   // Add good models from the active models to keep
   // and clear the active models
   modelCount=0;
   it = activeModels.begin();
   while( it != activeModels.end() && (*it).bic <= cutoffBIC && modelCount< gMaxKeptModels) {
    it++;
    modelCount++;
   }
   keepModels.insert(activeModels.begin(), it);
   if(keepModels.size() > gMaxKeptModels){
				it = keepModels.begin();
				modelCount=0;
		  while( it != keepModels.end() && modelCount< gMaxKeptModels) {
     it++;
     modelCount++;
				}
    cutoffBIC=it->bic;
				twoLogOR=cutoffBIC-minBIC;
   	keepModels.erase(it, keepModels.end());
			}	
   
   activeModels.clear();
   // Add good models from the next models to the active models
   // and clear the next models

   modelCount=0;it = nextModels.begin();				
   while( it != nextModels.end() && (*it).bic <= cutoffBIC && modelCount < gMaxActiveModels ) {
    it++;
    modelCount++;
   }
   activeModels.insert(nextModels.begin(), it);
   if(activeModels.size()== gMaxActiveModels){
				cutoffBIC=it->bic;
				twoLogOR=cutoffBIC-minBIC;

			}	
   nextModels.clear();

  }
  const double maxBIC = (double) keepModels.rbegin()->bic;
  memset(postProbs,0,nCols*sizeof(double));
  double postProbnorm=0;
  T retvalue=0;	  
  for ( it = keepModels.begin(); it != keepModels.end(); it++ ){
			const double postProb=exp((-0.5)*((double) it->bic-maxBIC));
			postProbnorm+=postProb;
			PackedBitArray<uint64_t> *list=it->modelIndices.list;
			const uint8_t modelBits=it->modelIndices.modelBits;
			const uint16_t nModels=it->modelIndices.nModels;
			for(int i=0;i<nModels;i++){
				postProbs[pord[list->get(modelBits,i)]]+=postProb;
			}
			T r2= it->r2;
			retvalue += postProb*((T) (nRows - nModels - 1) *log(1.0+g) - (T)(nRows-1)*log(1.0+g*(1.0-r2)));
		}	
		postProbnorm=(double)1/postProbnorm;
		int my_npostProbs=0;	
		T rvalue=0;
		if(parents){//sparse case - returns list of parents
   for (int i=0;i<nCols;i++){
		 	if(postProbs[i]){
		 	 postProbs[my_npostProbs]=postProbs[i]*postProbnorm;
		 	 parents[my_npostProbs++]=i;
		 	}
		 }
		}
		else{//dense case - returns vector of weights	
			for (int i=0;i<nCols;i++){
				if(postProbs[i]){
		   postProbs[i]=postProbs[i]*postProbnorm;
		   my_npostProbs++;
				}
			}
		}	
		*npostProbs=my_npostProbs;
		return(-retvalue);	
	}	

template <class T> void initRegressParms(T *A, T *ATA, T **data,int nGenes,int nRows, int nSamples,int nTimes,int &nVars,int nThreads,bool timeSeries ){

  //all the information is stored in the data array - data[gene_index][sample_index];
  //this order allows the column (gene) slices to be easil used
  //there are 100 genes and  582 rows - 6 time points per row
  // y vector t1-t5 xmatrix will be rowx t0-t4

  const int ATAldr=nGenes+1;
  const int Aldr=nRows;
  //column of ones to force calculation of intercepts
  for (int i=0;i<nRows;i++) A[i]=1;

		//copy variable columns
		if(timeSeries){
   T *dest=A+Aldr;
   for (int j=0;j<nGenes;j++){
				for (int n=0;n<nSamples;n++){
			  if(n%nTimes != nTimes-1){
						*dest=data[j][n];
						*dest++;
					}
		  }
   }
		}
		else{
   T *dest=A+Aldr;
   for (int j=0;j<nGenes;j++){
				for (int n=0;n<nRows;n++){
					*dest=data[j][n];
					dest++;
		  }
   }
		}
			//square the matrix;
	sqmm(nRows,nGenes+1,A,Aldr,ATA,ATAldr,nThreads);
	nVars=(nVars) ? nVars:nGenes;		
	initHashLUT(nVars);
	findModelBits(nVars);

}	
   
template <class T> int findRegulators(T g,int optimizeBits,int maxOptimizeCycles,float uPrior,float twoLogOR,int nVars,int nThreads,bool rankOnly,int geneIndex,T **data,T **rProbs,int *parents, double *postProbs, T *A, T *ATA, int Aldr, int ATAldr,int nGenes,int nRows,int nSamples,int nTimes){
	T *b=new T[nRows];
	T *ATb=new T[nGenes+1];
	T btb,sst;
	T *priors=new T[nGenes];
	if(rProbs && rProbs[geneIndex]){
		T* const slice = rProbs[geneIndex];
		for (int k=0;k<nGenes;k++){
			priors[k]=(T) slice[k];
		} 	 
	}
	else{
		for (int k=0;k<nGenes;k++)priors[k]=uPrior;
	}		
	
		//set up y vector
	if(nTimes){//TimeSeries case
		int j=0;
		for (int n=0;n<nSamples;n++){
			if(n%nTimes) b[j++]=data[geneIndex][n];
		}			 
	}
	else{ //nonTimeSeries case
		for (int n=0;n<nRows;n++){
			b[n]=data[geneIndex][n];
		}			
	}	

	//multiply AT b to get ATb
	mtrv(nRows,nGenes+1,A,Aldr,b,ATb);
 btb=dot(nRows,b,b);
 sst=calculate_sst(nRows,b,A); //A is passed because the first column is a column of 1s which is useful to find sum of y components using the dot product 
			//what to set twoLogOR and g to
	int ignoreIndex =(nTimes)? -1:geneIndex;
	int nEdges=fastScanBMA_g(ATA,ATb,btb,sst,ignoreIndex,priors,rankOnly,postProbs,parents,nRows, nGenes,nVars, twoLogOR,g,optimizeBits,maxOptimizeCycles); 
 //read return and delete each of the arrays as necessary			
	delete[] priors;
	delete[] b;
	delete[] ATb;
	return(nEdges);
}
   

template <class T> T calculate_sst(int sizeb, T *b,T *ones){      
 //calculate sst which is used to calculate the R2
  T bbar=dot(sizeb,b,ones)/(T)sizeb; //first column of A contains row of ones 
  T *temp=new T[sizeb];
  memmove(temp,b,sizeb*sizeof(T));
  axpy(sizeb,-bbar,ones,temp);
  T sst=dot(sizeb,temp,temp);
  delete[] temp;
  return(sst);
}
template <class T> T getR2_full(ModelIndices &modelIndices, T* ATA,int ATAldr, T* ATb,const T btb,T *R,int Rldr){
	//returns the R2 value of the fit
	//writes the cholesky decomposition to R (without the best fit coordinates);
 const int p=modelIndices.nModels+1; 
 //add 1 because of 1s column - this is the dimension of the square Cholesky submatrix
 //However we also add a column/row for (ATb,btb) vector - trick to get the ssq/n and betahats  
	memset(R,0,(p+1)*Rldr*sizeof(T));
 for (int i=0;i<p;i++){
			//column 0 corresponds to column of 1s in ATA and in Rarray
			//column n corresponds to variable n-1 -> add one to this to get corresponding column in ATA matrix (because of 1s col...)
			//last column will be vector b
			//rightmost diagonal will b btb
			//solving this gives betahat and ssq
		const int coli= (i)?  modelIndices.list[i-1]+1:0;
		T* const Rslice=R+(i*Rldr);
		T* const ATAslice=ATA+(coli*ATAldr);
		for (int j=0;j<=i;j++){
			const int colj= (j)? modelIndices.list[j-1]+1 : 0;	
			Rslice[j]=ATAslice[colj];
		}
	}
	//last colum
	T* const b=R+(p*Rldr);
 for (int i=0;i<p;i++){
		const int col= (i)?  modelIndices.list[i-1]+1:0;
		b[i]=ATb[col];
	}
 b[p]=btb;
 potrf('U',p+1,R,p+1);
 T ssqn1= R[(p*Rldr)+p];
 return(ssqn1*ssqn1);
} 
template <class T> T getR2_up(ModelIndices &modelIndices,T* ATA,int ATAldr, T* ATb,const T btb,T *R,int Rldr){
	//first solve for new triangular matrix using trsv
	//then solve for R_orig using new matrix trsv
	//writes over extra column in R_orig
	double rho;
	const int nRows=modelIndices.nModels+1; //the modelIndices include the added model - add one to this because of 1's column
	const int nCols=nRows-1;
 const int modelCol=modelIndices.list[modelIndices.nModels-1]+1;

	//v last column of R
	T* const v=R+nCols*Rldr;
	T* const ATAslice=ATA+(ATAldr*modelCol);
	v[0]=ATAslice[0];//1s column term
	for(int i=1;i<nRows;i++){
		const int colj=modelIndices.list[i-1]+1;
		v[i]=ATAslice[colj];
	}
 
 T temp=v[nRows-1];
 //calculate new vector to add to matrix using triangular solver
 trsvutr(nCols,R,Rldr,v); 
 rho=dot(nCols,v,v); 
 v[nRows-1]=(temp-rho>0)? sqrt	(temp-rho): 0;
 
 //add new response vector as column
 T *w=new T[nCols+1];
 w[0]=ATb[0]; //for the column of ones
 for (int i = 0; i <nCols; i++){
		w[i+1]=ATb[modelIndices.list[i]+1];
	}

 //number of columns have increased by 1...
 trsvutr(nCols+1,R,Rldr,w);
 rho=dot(nCols+1,w,w);
 /*
 if(isinff(rho) || isnanf(rho)){
  //throw the variable away - it is either near zero or an a multiple of another variable
  delete[] w;
  return(btb);
	}
 */
	
 delete[] w;
 return(btb-rho);
}
template <class T> T getR2_down(int nRows,ModelIndices &modelIndices,T* ATb,const T btb,T *R, int Rldr, int dCol){	
	//dCol is deleted column number
	const int nCols=nRows-1;
 //retriangularize the rows starting at the jth row
 //two workspace vectors required to store c and s for givens transforms
 //we use last column for one of these vectors

 const int size=(nRows-dCol-1<nCols-dCol)?nRows-dCol-1:nCols-dCol;
 T c[size],s[size],v[nCols];
 qhqr(nRows-dCol,nCols-dCol,&(R[dCol*Rldr+dCol]),Rldr,c,s);

	//calculate response vector
 v[0]=ATb[0]; //corresponds to column of 1s
 int k=1;
 for (int i = 0; i <dCol-1; i++){ 
	 v[k++]=ATb[modelIndices.list[i]+1];
	} 
	for (int i = dCol; i <nCols; i++){
	 v[k++]=ATb[modelIndices.list[i]+1]; 
	}
	T rho;
	//back substitute to get residuals
 trsvutr(nCols,R,Rldr,v);
 rho=dot(nCols,v,v);
  if(!isfinite(rho)){
  //throw the variable away - it is either near zero or an a multiple of another variable
  return(btb);
	}
 return(btb-rho);  
}		

//extra routines for update/downdate
template <class T> void qhqr (int nRows,int nCols,T *R, int ldr,T *c,T *s){
 //c++ implmentation of routines from qrupdate 
 //same routine except the pointer stuff is all explicit
 if (nRows <1 || nCols == 0) return;
 T* Ri=R;//pointer to ith column start
 for(int i=0;i<nCols;i++){
  //apply stored rotations, column-wise
  T t = Ri[0];
  int ii=(i<nRows-1)? i : nRows-1;
  for (int j=0;j<ii;j++){
   Ri[j]=c[j]*t+s[j]*Ri[j+1];
   t=c[j]*Ri[j+1]-s[j]*t; 
		}      
  if (ii < nRows-1){
			lartg(&t,&(Ri[ii+1]),&(c[i]),&(s[i]),&(Ri[ii])); //calculates the rotations - blas rotg does not work 
   Ri[ii+1]=0;
		}
		else{
			Ri[ii]=t;
		}
		Ri+=ldr;
	}
}	


//routines are now wrapped to interface directly with BLAS routines
//disable double precsion

//wrappers for cblas routines -see predeclarations for description
#ifdef USE_RBLAS

 void sqmm(int nRows,int nCols,double *A,int Aldr, double *ATA,int ATAldr,int nThreads){
		double one=1, zero=0;	
 	dgemm_("T","N",&nCols,&nCols,&nRows,&one,A,&Aldr,A,&Aldr,&zero,ATA,&ATAldr);
 }	
 void mtrv(int nRows, int nCols, double *A, int Aldr, double *b, double *ATb){
  double done=1,dzero;
  int ione=1,izero=0;
 	dgemv_("T",&nRows,&nCols,&done,A,&Aldr,b,&ione,&dzero,ATb,&ione);
 }
 void lartg (double* f, double* g, double* cs, double* sn, double* r){
		dlartg_(f,g,cs,sn,r);
	}	
 double dot(int n,double *x,double *y){
		int one=1;	
		return(ddot_(&n,x,&one,y,&one));
	}
 void axpy (int n,double a,double *x,double *y){
		int ione=1;
		daxpy_(&n,&a,x,&ione,y,&ione);
	}	
 void trsvutr(int n,double *R, int Rldr, double *v){
		int ione=1;
	 dtrsv_("U","T","N",&n,R,&Rldr,v,&ione); 
	}

#else
void sqmm(int nRows,int nCols,float *A,int Aldr, float *ATA,int ATAldr){cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, nCols,nCols,nRows,1,A,Aldr,A,Aldr,0,ATA,ATAldr);}	
void sqmm(int nRows,int nCols,double *A,int Aldr, double *ATA,int ATAldr){cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nCols,nCols,nRows,1,A,Aldr,A,Aldr,0,ATA,ATAldr);}
void sqmm(int nRows,int nCols,float *A,int Aldr, float *ATA,int ATAldr,int nThreads){
	cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, nCols,nCols,nRows,1,A,Aldr,A,Aldr,0,ATA,ATAldr);
	}	
void sqmm(int nRows,int nCols,double *A,int Aldr, double *ATA,int ATAldr,int nThreads){
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nCols,nCols,nRows,1,A,Aldr,A,Aldr,0,ATA,ATAldr);
	}	
void mtrv(int nRows, int nCols, double *A, int Aldr, double *b, double *ATb){cblas_dgemv(CblasColMajor, CblasTrans, nRows,nCols,1,A,Aldr,b,1,0,ATb,1);}
void mtrv(int nRows, int nCols, float *A, int Aldr, float *b, float *ATb){cblas_sgemv(CblasColMajor, CblasTrans, nRows,nCols,1,A,Aldr,b,1,0,ATb,1);  }
void lartg (double* f, double* g, double* cs, double* sn, double* r){dlartg_(f,g,cs,sn,r);}	
void lartg (float* f, float* g, float* cs, float* sn, float* r){slartg_( f,  g,  cs,  sn,  r);}
double dot(int n,double *x,double *y){return(cblas_ddot(n,x,1,y,1));}
float dot(int n,float *x,float *y){return(cblas_sdot(n,x,1,y,1));}	
void axpy (int n,double a,double *x,double *y){cblas_daxpy(n,a,x,1,y,1);}	
void axpy (int n,float a,float *x,float *y){cblas_saxpy(n,a,x,1,y,1);}
void trsvutr(int n,float *R, int Rldr, float *v){cblas_strsv(CblasColMajor,CblasUpper,CblasTrans,CblasNonUnit,n,R,Rldr,v,1); }	
void trsvutr(int n,double *R, int Rldr, double *v){cblas_dtrsv(CblasColMajor,CblasUpper,CblasTrans,CblasNonUnit,n,R,Rldr,v,1); }
#endif

void potrf(char ul,int n,double *R,int Rldr){
 char uplo=ul;
 int m=n;
 int lda=Rldr;
 int info;
 LAPACK_dpotrf(&uplo,&m,R,&lda,&info);
}
#ifndef USE_RBLAS 
void potrf(char ul ,int n,float *R,int Rldr){
 char uplo=ul;
 int m=n;
 int lda=Rldr;
 int info;
 LAPACK_spotrf(&uplo,&m,R,&lda,&info);
}
#endif
//these routines are necessary as overloading the = operator requires that both ModelIndices and Const/CompactIndices be predeclared which requires wrapper class or redoing this as shared/inherited classes
template <class T> void copy_indices(ModelIndices &dest,const CompactModelIndices<T>  &source){
	dest.nModels=source.nModels;
	const uint8_t modelBits=source.modelBits;
	if(source.nModels){
		for(int i=0;i<source.nModels;i++){
			dest.list[i]=source.list->get(modelBits,i);
		}	
	}
	memset(dest.index,0,dest.maxModels*sizeof(uint16_t));	
 for(uint16_t i=0;i<dest.nModels;i++) dest.index[dest.list[i]]=i+1;
  dest.hashValue=source.hashValue;
}
template <class T> T TimeSeriesValuesToResiduals(T *values, T *residuals,int nTimes,int nGroups){
	
	//given a set of time series replicates this performs the following
	//the mean at each time over all replicates with that time is calculated 
	//expression values at each time are transformed by subtracting the mean
	//a linear model is constructed for each vector of these values at time t from the values at t-1
	//i.e. values_t= A*values_t_previous + B;
	//returns the ssq/n
	
	//input is all values of a specific time value
	//assume all replicates have the same time unit
	//assume that all times are already pre-sorted - with first element being low
	
	//calculate means over replicate
	
	vector <T> sums;
	sums.resize(nTimes);
	for(int i=0;i<nTimes;i++)
	 sums[i]=0;

 const int nSamples=nGroups*nTimes;
	for(int i=0;i<nSamples;i++){
	 sums[i%nTimes]+=values[i];
	}
	for (int i=0;i<nTimes;i++){
	 sums[i]/= (T) nGroups;
 }
 
 //subtract means from values
 for(int i=0;i<nSamples;i++){
		values[i]-=sums[i%nTimes];
	}	

 //set up cholesky decomp  
 
 const int nRows =nGroups*(nTimes-1);
 
	T *A=new T [2*nRows];
	T ATA[4];//2x2 matrix
	T R[9];//3x3 matrix 
	T *b=new T[nRows];
 const int ATAldr=2;
 const int Aldr=nRows;
 
 //set up data matrix
 //columnmajor so columns first...
 //first column is 1's 
 for (int i=0;i<nRows;i++)A[i]=1;	 
 for (int i=nRows;i<2*nRows;i++)A[i]=values[i-nRows];
 //set up response vector i.e. values at time t
 for (int i=0;i<nRows;i++)b[i]=values[i+nGroups];
 
	//square the data matrix;
	sqmm(nRows,2,A,Aldr,ATA,ATAldr,1); 
	
	T ATb[2];
	
	mtrv(nRows,2,A,Aldr,b,ATb);
 T btb=dot(nRows,b,b);
	
	//copy ATA to R matrix for Cholesky with last column ATb and btb which will be transformed into ATbeta and b*b
	R[0]=ATA[0];
	R[1]=ATA[1];
	R[2]=0;
	R[3]=ATA[2];
	R[4]=ATA[3];
	R[5]=0;
	R[6]=ATb[0]; //betahat for first column of ones i.e. the intercept
	R[7]=ATb[1]; //betahat for second column - values - i.e. the slope
	R[8]=btb;

 potrf('U',3,R,3);


 //back substitute to get residuals
 trsvutr(2,R,3,R+6);

 
 const T slope=R[7];
 const T intercept=R[6];
 
 //decide whether there is padding or not
 for (int i=0;i<nRows;i++){
	 residuals[i]=values[i+nGroups]-values[i]*slope-intercept;
	}
	delete [] A;
	delete [] b;
	return(R[8]);	
}














