// MainFrm.cpp : implementation of the CMainFrame class
//

#include "stdafx.h"
#include "GxViewer.h"
#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
	//{{AFX_MSG_MAP(CMainFrame)
	ON_COMMAND(ID_OPEN_DEVICE, OnOpenDevice)
	ON_COMMAND(ID_START_SNAP, OnStartSnap)
	ON_COMMAND(ID_STOP_SNAP, OnStopSnap)
	ON_COMMAND(ID_CLOSE_DEVICE, OnCloseDevice)
	ON_WM_CLOSE()
	ON_UPDATE_COMMAND_UI(ID_OPEN_DEVICE, OnUpdateOpenDevice)
	ON_UPDATE_COMMAND_UI(ID_START_SNAP, OnUpdateStartSnap)
	ON_UPDATE_COMMAND_UI(ID_STOP_SNAP, OnUpdateStopSnap)
	ON_UPDATE_COMMAND_UI(ID_CLOSE_DEVICE, OnUpdateCloseDevice)
	ON_WM_PAINT()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
	// TODO: add member initialization code here
	m_hDC                   = NULL;            // The handle for DC                    
	m_hDevice               = NULL;            // The handle for device 
	m_pBmpInfo              = NULL;	           // The pointer for showing image 
	m_pRawBuf               = NULL;            // The pointer for showing raw image 
	m_pShowBuf              = NULL;            // The pointer for showing buffer
	m_nImageWidth           = 0;               // Acquired ImageWidth
	m_nImageHeight          = 0;               // Acquired ImageHeight
	m_nPayLoadSize          = 0;               // Acquired raw image size
	m_nPixelColorFilter     = 0;               // PixelColorFilter 
	m_bIsColorFilter        = false;           // Check whether the device supports PixelColorFilter
	m_bIsOpen               = false;           // The flag indicates whether the first device has been opened
	m_bIsSnap               = false;           // The flag indicates whether AcquisitionStart command has been send 

	// Before using any GxIAPI methods, the GxIAPI must be initialized.   
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}

	memset(m_chBmpBuf, 0, sizeof(m_chBmpBuf));
}

CMainFrame::~CMainFrame()
{
	// Release GxIAPI  
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	emStatus = GXCloseLib();
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CFrameWnd::PreCreateWindow(cs) )
		return FALSE;
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return TRUE;
}

/////////////////////////////////////////////////////////////////////////////
// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWnd::Dump(dc);
}

#endif //_DEBUG

void CMainFrame::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	
	// TODO: Add your message handler code here
	CString strTitle;
	strTitle.Format("GxViewer");
	SetWindowText(strTitle);
	// Do not call CFrameWnd::OnPaint() for painting messages
}

/////////////////////////////////////////////////////////////////////////////
// CMainFrame message handlers

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close the application.


\return  void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
    GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send AcquisitionStop commands
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	
		// Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		
		// Release buffers for image acquisition
		UnPrepareForShowImage();	
		m_bIsSnap = false;
	}

	// If the device has been opened then close it.
	if (m_bIsOpen)
	{
		emStatus = GXCloseDevice(m_hDevice);
		m_bIsOpen = false;
	}		
	CFrameWnd::OnClose();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnOpenDevice() 
{
	// TODO: Add your command handler code here
	GX_STATUS      emStatus   = GX_STATUS_SUCCESS;
	uint32_t       nDeviceNum = 0;
	GX_OPEN_PARAM  stOpenParam;

	// Enumerate all the camera devices 
	emStatus = GXUpdateDeviceList(&nDeviceNum, 1000);
	GX_VERIFY(emStatus);

	// Check if found any device
	if (nDeviceNum <= 0)
	{
		MessageBox("No devices found!");
		return;
	}

	// If the device is opened then close it to ensure the camera could open again.
	if (m_hDevice != NULL)
	{
		emStatus = GXCloseDevice(m_hDevice);
		GX_VERIFY(emStatus);
		m_hDevice = NULL;
	}

	// Open the first found device in the list  
	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_INDEX;
	stOpenParam.pszContent = "1";
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	GX_VERIFY(emStatus);

    // It is suggested that users set the camera's packet size according to the current network environment after opening the network camera.
    // To improve the acquisition performance of network cameras, the following code is used for reference.
    {
        bool	 bImplementPacketSize = false;
        uint32_t unPacketSize		  = 0;

        emStatus = GXIsImplemented(m_hDevice, GX_INT_GEV_PACKETSIZE, &bImplementPacketSize);
        GX_VERIFY(emStatus);

        if (bImplementPacketSize)
        {
            emStatus = GXGetOptimalPacketSize(m_hDevice,&unPacketSize);
            GX_VERIFY(emStatus);

            emStatus = GXSetInt(m_hDevice, GX_INT_GEV_PACKETSIZE, unPacketSize);
            GX_VERIFY(emStatus);
        }
    }


	emStatus = InitParam();
	GX_VERIFY(emStatus);
	m_bIsOpen = true;	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to start acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnStartSnap() 
{
	// TODO: Add your command handler code here
	bool      bReturn  = false;
    GX_STATUS emStatus = GX_STATUS_SUCCESS;

	bReturn  = PrepareForShowImage();
	if (!bReturn)
	{
		MessageBox("Fail to allocate resource for showing image!");
		return;
	}

	// Register the CaptureCallback function 
	emStatus = GXRegisterCaptureCallback(m_hDevice, this, OnFrameCallbackFun);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImage();
		ShowErrorString(emStatus);
		return;
	}

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevice, GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImage();
		ShowErrorString(emStatus);
		return;
	}

	// Send AcquisitionStart Command 
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImage();
		ShowErrorString(emStatus);
		return;
	}
	m_bIsSnap = true;	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnStopSnap() 
{
	// TODO: Add your command handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Send AcquisitionStop Command 
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);
	
	// Unregister the CaptureCallback function
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	GX_VERIFY(emStatus);
    
	// Release memory for showing image
	UnPrepareForShowImage();
	m_bIsSnap = false;	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnCloseDevice() 
{
	// TODO: Add your command handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send AcquisitionStop Command 
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here 
		}
		
		// Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here 
		}
		
		// Release memory for showing image
		UnPrepareForShowImage();
		m_bIsSnap = false;
	}
	
	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}
	m_hDevice = NULL;
	m_bIsOpen = false;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CMainFrame::InitParam()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerMode off
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);	
	VERIFY_STATUS_RET(emStatus);

	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8Bit(); 
	VERIFY_STATUS_RET(emStatus);

	// Get the PayLoadSize 
	emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_nPayLoadSize);
	VERIFY_STATUS_RET(emStatus);
	
	// Get ImageWidth
	emStatus = GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth);
	VERIFY_STATUS_RET(emStatus);
	
	// Get ImageHeight
	emStatus = GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);
	VERIFY_STATUS_RET(emStatus);
	
	// Check whether the device supports PixelColorFilter
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_bIsColorFilter);
	VERIFY_STATUS_RET(emStatus);
	
	// Get the PixelColorFilter 
	if(m_bIsColorFilter)
	{
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter);
	}

	return emStatus;
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit 

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CMainFrame::SetPixelFormat8Bit()
{
	uint32_t  i           = 0;
	int64_t   nPixelSize  = 0;                     
	uint32_t  nEnmuEntry  = 0;
	size_t    nBufferSize = 0;	
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	GX_ENUM_DESCRIPTION  *pEnumDescription = NULL;
	GX_ENUM_DESCRIPTION  *pEnumTemp        = NULL;
	
	// Get the feature PixelSize, this feature indicates the depth of the pixel values in the acquired images in bits per pixel
	emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_SIZE, &nPixelSize);
	VERIFY_STATUS_RET(emStatus);
	
	// If the size is 8bit then return
	if (nPixelSize == GX_PIXEL_SIZE_BPP8)
	{
		return GX_STATUS_SUCCESS;
	}
	
	// Get the enumeration entry of the pixel format the device supports.
	emStatus = GXGetEnumEntryNums(m_hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
	VERIFY_STATUS_RET(emStatus);
		
	// Allocate memory for getting the enumeration entry of the pixel format.
	nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
	pEnumDescription = new GX_ENUM_DESCRIPTION[nEnmuEntry];
		
	emStatus = GXGetEnumDescription(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// Release memory
		if (pEnumDescription != NULL)
		{
			delete []pEnumDescription;
			pEnumDescription = NULL;
		}
		return emStatus;
	}
			
	// A loop to visit every enumeration entry node once to find which enumeration entry node is 8bit and then set pixel format to its value 
	pEnumTemp = pEnumDescription;
	for (i = 0; i < nEnmuEntry; i++)
	{
		if ((pEnumTemp->nValue & GX_PIXEL_8BIT) == GX_PIXEL_8BIT)
		{
			emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumTemp->nValue);
			break;
		}
		pEnumTemp++;
	}	
		
	// Release memory
	if (pEnumDescription != NULL)
	{
		delete []pEnumDescription;
		pEnumDescription = NULL;
	}
	
	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to allocate memory for showing color image.
\return  true: Success false: Fail
*/
//----------------------------------------------------------------------------------
bool CMainFrame::PrepareForShowImage()
{	
	int i = 0;
	size_t nImageSize = 0;

	m_hDC        = ::GetDC(m_hWnd);

	// Initialize bitmap header
	m_pBmpInfo						  	  = (BITMAPINFO *)m_chBmpBuf;
	m_pBmpInfo->bmiHeader.biSize		  = sizeof(BITMAPINFOHEADER);
	m_pBmpInfo->bmiHeader.biWidth		  = (LONG)m_nImageWidth;
	m_pBmpInfo->bmiHeader.biHeight		  = (LONG)m_nImageHeight;	
	
	m_pBmpInfo->bmiHeader.biPlanes		  = 1;
	m_pBmpInfo->bmiHeader.biBitCount	  = m_bIsColorFilter ? 24 : 8; 
	m_pBmpInfo->bmiHeader.biCompression	  = BI_RGB;
	m_pBmpInfo->bmiHeader.biSizeImage	  = 0;
	m_pBmpInfo->bmiHeader.biXPelsPerMeter = 0;
	m_pBmpInfo->bmiHeader.biYPelsPerMeter = 0;
	m_pBmpInfo->bmiHeader.biClrUsed		  = 0;
	m_pBmpInfo->bmiHeader.biClrImportant  = 0;

	// If using mono cameras,you need to initialize color palette first.
	if (!m_bIsColorFilter)
	{
		for(i = 0; i < 256; i++)
		{
			m_pBmpInfo->bmiColors[i].rgbBlue	 = i;
			m_pBmpInfo->bmiColors[i].rgbGreen	 = i;
			m_pBmpInfo->bmiColors[i].rgbRed		 = i;
			m_pBmpInfo->bmiColors[i].rgbReserved = i;
		}
	}
	
	// Allocate memory for showing raw images.
	m_pRawBuf = new BYTE[(size_t)(m_nPayLoadSize)];
	if (m_pRawBuf == NULL)
	{
		return false;
	}

	if (m_bIsColorFilter)   
	{
		nImageSize = (size_t)(m_nImageHeight * m_nImageWidth * 3);
	}
	else                    
	{
		nImageSize = (size_t)(m_nImageHeight * m_nImageWidth);
	}

	// Allocate memory for showing image
	m_pShowBuf = new BYTE[nImageSize];
	if (m_pShowBuf == NULL)
	{
		if (m_pRawBuf != NULL )
		{
			delete []m_pRawBuf;
			m_pRawBuf = NULL;
		}			
		return false;
	}
	
	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release the resources for showing image 

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::UnPrepareForShowImage()
{
	::ReleaseDC(m_hWnd,m_hDC);

	if (m_pRawBuf != NULL)
	{
		delete []m_pRawBuf;
		m_pRawBuf = NULL;
	}
	
	if (m_pShowBuf != NULL)
	{
		delete []m_pShowBuf;
		m_pShowBuf = NULL;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to convert the image format and draw image
\param   pFrame 

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CMainFrame::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	VxUint32   i       = 0;
	CMainFrame *pDlg   = (CMainFrame*)(pFrame->pUserParam);
	VxUint32   nHeight = (VxUint32)pDlg->m_nImageHeight;
	VxUint32   nWidth  = (VxUint32)pDlg->m_nImageWidth;

	if (pFrame->status != 0)
	{
		return;
	}
    
	// Copy the image data from callback function
	memcpy(pDlg->m_pRawBuf, pFrame->pImgBuf, pFrame->nImgSize);
	
	if (pDlg->m_bIsColorFilter)
	{
		// If the acquired image is color format,convert it to RGB 
		DxRaw8toRGB24(pDlg->m_pRawBuf, pDlg->m_pShowBuf, nWidth, nHeight, RAW2RGB_NEIGHBOUR, 
			          DX_PIXEL_COLOR_FILTER(pDlg->m_nPixelColorFilter), TRUE);
	}
	else
	{
		// If the acquired image is mono format,you must flip the image data for showing.
		for(i = 0; i < nHeight; i++)
		{
			memcpy(pDlg->m_pShowBuf + i * nWidth, pDlg->m_pRawBuf + (nHeight - i - 1) * nWidth, nWidth);
		}		
	}

	pDlg->DrawImage(pDlg->m_pShowBuf, nWidth, nHeight, pDlg->m_pBmpInfo);	
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to draw image in the window
\param   pImageBuf   
\param   nWidth    
\param   nHeight   
\param   pBmpInfo		

\return  void
*/
//----------------------------------------------------------------------------------
void CMainFrame::DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo)
{
	if ((pImageBuf == NULL) || (pBmpInfo == NULL))
	{
		return;
	}

	RECT  objRect; 
	::GetWindowRect(m_hWnd, &objRect);
	int nWndWidth  = objRect.right  - objRect.left;
	int nWndHeight = objRect.bottom - objRect.top;
	
	// You could use the following code to show image. 
	::SetStretchBltMode(m_hDC, COLORONCOLOR);
	::StretchDIBits(m_hDC,
		0,						
		0,
		nWndWidth,
		nWndHeight,
		0,
		0,
		nWidth,
		nHeight,
		pImageBuf,
		pBmpInfo,
		DIB_RGB_COLORS,
		SRCCOPY);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::ShowErrorString(GX_STATUS emErrorStatus)
{
	char      *pchErrorInfo = NULL;
	size_t    nSize         = 0;
	GX_STATUS emStatus      = GX_STATUS_ERROR;

	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (pchErrorInfo == NULL)
	{
		return;
	}
	
	// Get error information and show them to user 
	emStatus = GXGetLastError(&emErrorStatus, pchErrorInfo, &nSize);

	if (emStatus != GX_STATUS_SUCCESS)
	{
		MessageBox("Fail to call GXGetLastError interface!");
	}
	else
	{
		MessageBox((LPCTSTR)pchErrorInfo);
	}

	// Release memory 
	if (pchErrorInfo != NULL)
	{
		delete[]pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

void CMainFrame::OnUpdateOpenDevice(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if (!m_bIsOpen)
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}	
}

void CMainFrame::OnUpdateStartSnap(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if (m_bIsOpen && (!m_bIsSnap))
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}	
}

void CMainFrame::OnUpdateStopSnap(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if (m_bIsOpen && m_bIsSnap)
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}	
}

void CMainFrame::OnUpdateCloseDevice(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if (m_bIsOpen)
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}	
}

