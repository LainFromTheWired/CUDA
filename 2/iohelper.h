#ifndef _helper_h
#define _helper_h

#include <assert.h>
#include <stdio.h>

template <class T>
class IOHelper 
{
public:
	IOHelper();

	void WriteData(T *data);

	int ReadData(T *data);

	int GetWeight();
	int GetHeight();
	int GetRadius();

	~IOHelper();
  
private: 
	char input_name[256];  
	char output_name[256];  
 
	int radius, weight, hieght;
 
	FILE *input;
	FILE *output;	
};


#include "iohelper.cu"

#endif