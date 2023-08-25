#ifndef GX_EX_H
#define GX_EX_H
#include "GxIAPI.h"

typedef struct CAMER_INFO
{
	BITMAPINFO					*pBmpInfo;		  ///< The buffer for showing image
	BYTE						*pImageBuffer;	  ///< The pointer for the image has been processed 
	BYTE                        *pRawBuffer;      ///< The pointer for the raw data of image
	char						chBmpBuf[2048];	  ///< The buffer for BMP image
	int64_t                     nPayLoadSise;     ///< Image payload size
	int64_t					    nImageWidth;	  ///< ImageWidth
	int64_t					    nImageHeight;	  ///< ImageHeight
	int64_t					    nBayerLayout;	  ///< PixelColorFilter
	bool						bIsColorFilter;	  ///< The flag indicates Whether the acquired image is color filter
	BOOL						bIsOpen;		  ///< The flag indicates whether the first device has been opened
	BOOL						bIsSnap;		  ///< The flag indicates whether image acquisition has started
	float						fFps;			  ///< Fps
}CAMER_INFO;
	
#endif