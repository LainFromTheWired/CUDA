#pragma once
#include "iohelper.h"


template <class T>
IOHelper<T>::IOHelper() 
{
	scanf("%s", input_name);
	scanf("%s", output_name);
	scanf("%d", &radius);

	input = fopen(input_name, "rb");		

	fread(&weight, sizeof(weight), 1, input);
	fread(&hieght, sizeof(hieght), 1, input);
};


template <class T>
void IOHelper<T>::WriteData(T *data)
{	
	output = fopen(output_name, "wb");

	fwrite(&weight, sizeof(weight), 1, output);
	fwrite(&hieght, sizeof(hieght), 1, output);
	fwrite(data, sizeof(*data), weight * hieght, output);	
};

template <class T>
int IOHelper<T>::ReadData(T *data) 
{
	return fread(data, sizeof(*data), weight * hieght, input);	
};


template <class T>
int IOHelper<T>::GetWeight() 
{
	return weight;
};

template <class T>
int IOHelper<T>::GetHeight()
{
	return hieght;
};

template <class T>
int IOHelper<T>::GetRadius()
{
	return radius;
};

template <class T>
IOHelper<T>::~IOHelper() 
{  
	fclose(input);
	fclose(output); 
};      
