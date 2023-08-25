// MainFrm.cpp : implementation of the CMainFrame class
//

#include "stdafx.h"
#include "GxMultiCam.h"
#include "MainFrm.h"
#include "CamerParamDlg.h"
#include "SltCamDlg.h"
#include "GxMultiCamDOC.h"
#include "GxMultiCamVIEW.h"

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
	ON_WM_CREATE()
	ON_COMMAND(ID_MENU_SLTCAM, OnMenuSltcam)
	ON_COMMAND(ID_MENU_OPEN, OnMenuOpen)
	ON_COMMAND(ID_MENU_CLOSE, OnMenuClose)
	ON_COMMAND(ID_MENU_START, OnMenuStart)
	ON_WM_SIZE()
	ON_WM_TIMER()
	ON_COMMAND(ID_MENU_STOP, OnMenuStop)
	ON_UPDATE_COMMAND_UI(ID_MENU_OPEN, OnUpdateMenuOpen)
	ON_UPDATE_COMMAND_UI(ID_MENU_START, OnUpdateMenuStart)
	ON_UPDATE_COMMAND_UI(ID_MENU_CLOSE, OnUpdateMenuClose)
	ON_UPDATE_COMMAND_UI(ID_MENU_STOP, OnUpdateMenuStop)
	ON_COMMAND(ID_MENU_CAMID, OnMenuCamid)
	ON_UPDATE_COMMAND_UI(ID_MENU_CAMID, OnUpdateMenuCamid)
	ON_WM_CLOSE()
	ON_COMMAND(ID_MENU_CAMMER_PARAM, OnMenuCammerParam)
	//}}AFX_MSG_MAP
	ON_WM_PAINT()
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // status line indicator
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

/////////////////////////////////////////////////////////////////////////////
// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
	// TODO: add member initialization code here
	m_hDevices	 = NULL;									//The handle for device  
	m_pstCam	 = NULL;									//The structure for image data 
	m_nCamNum	 = 0;										//The number of connected cameras
	m_bViewID 	 = FALSE;									//The flag indicates whether show the deviceID 
	m_pBaseinfo	 = NULL;									//The pointer for initializing information of devices
	m_nOperateID = -1;                                      //The index of the selected camera
	m_pCamsFps   = NULL;                                    //Initializes the device Fps count object

	for (int i = 0; i < 4; i++)
	{
		m_pCWnds[i] = NULL;
		m_hDCs[i]   = NULL;
	}

	// Before using any GxIAPI methods, the GxIAPI must be initialized.  
	GX_STATUS emStatus = GX_STATUS_ERROR;
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}
}

CMainFrame::~CMainFrame()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;

	//Release memory
	if(m_pBaseinfo != NULL)
	{
		delete[]m_pBaseinfo;
		m_pBaseinfo = NULL;
	}
	if(m_hDevices != NULL)
	{
		delete[]m_hDevices;
		m_hDevices = NULL;
	}
	if (m_pCamsFps != NULL)
	{
		delete[] m_pCamsFps;
		m_pCamsFps = NULL;
	}
	if(m_pstCam != NULL)
	{
		for(uint32_t i = 0; i < m_nCamNum; i++)
		{
			if(m_pstCam[i].pImageBuffer != NULL)
			{
				delete[]m_pstCam[i].pImageBuffer;
				m_pstCam[i].pImageBuffer = NULL;
			}
			if (m_pstCam[i].pRawBuffer != NULL)
			{
				delete []m_pstCam[i].pRawBuffer;
				m_pstCam[i].pRawBuffer = NULL;
			}
		}
		delete[]m_pstCam;
        m_pstCam = NULL;
	}

	//Release GxIAPI 
	emStatus = GXCloseLib();
}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
	{
		return -1;
	}

	m_objMenu.LoadMenu(IDR_MENU_CAM);   
	this->SetMenu(&m_objMenu);          
	CMenu* pmSub = m_objMenu.GetSubMenu(0); 
	m_objMenu.EnableMenuItem(1, MF_BYPOSITION | MF_DISABLED | MF_GRAYED);
	SetTimer(0, 1000, NULL);
	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CFrameWnd::PreCreateWindow(cs))
	{
		return FALSE;
	}
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs
	m_strTitle = "GxMultiCam";
	cs.style &= ~FWS_ADDTOTITLE;
	cs.style &= ~WS_THICKFRAME;
	cs.style &= ~WS_MAXIMIZEBOX;

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

/////////////////////////////////////////////////////////////////////////////
// CMainFrame message handlers
//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to select the camera

\return  void
*/
//---------------------------------------------------------------------------------
void CMainFrame::OnMenuSltcam() 
{
	// TODO: Add your command handler code here
	//Call the dialog to select camera
	CSltCamDlg		m_SltDlg;			
	INT_PTR         nID  = m_SltDlg.DoModal(); 
	if (nID == IDCANCEL)
	{
		return;
	}
	
	if(m_nOperateID < 0)
	{
		return;
	}

	//Check whether the device is opened
	if(m_pstCam[m_nOperateID].bIsOpen)
	{
		//Enable the second menu if the device is opened
		m_objMenu.EnableMenuItem(1, MF_BYPOSITION | MF_ENABLED);
		SetMenu(&m_objMenu);
	}
	else
	{
		//Disable the second menu if the device is closed
		m_objMenu.EnableMenuItem(1, MF_BYPOSITION | MF_DISABLED | MF_GRAYED);
		SetMenu(&m_objMenu);
	}	
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to open device 

\return  void
*/
//---------------------------------------------------------------------------------
void CMainFrame::OnMenuOpen() 
{
	// TODO: Add your command handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	int64_t   nValue   = 0;
	bool      bIsImplemented = false;
	char      szIndex[10]    = {0};

    GX_OPEN_PARAM stOpenParam;
    stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
    stOpenParam.openMode   = GX_OPEN_INDEX;
    stOpenParam.pszContent = "";

	//The device number starts with 1
    _itoa(m_nOperateID + 1, szIndex, 10);

	// If the device is acquiring image then close it.
	if (m_hDevices[m_nOperateID] != NULL)
	{
		emStatus = GXCloseDevice(m_hDevices[m_nOperateID]);
		GX_VERIFY(emStatus);
		m_hDevices[m_nOperateID] = NULL;
	}


	stOpenParam.pszContent = szIndex;
	SetWindowText("The device is opening...");
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevices[m_nOperateID]);

    // It is suggested that users set the camera's packet size according to the current network environment after opening the network camera.
    // To improve the acquisition performance of network cameras, the following code is used for reference.
    {
        bool	 bImplementPacketSize = false;
        uint32_t unPacketSize		  = 0;

        emStatus = GXIsImplemented(m_hDevices[m_nOperateID], GX_INT_GEV_PACKETSIZE, &bImplementPacketSize);
        GX_VERIFY(emStatus);

        if (bImplementPacketSize)
        {
            emStatus = GXGetOptimalPacketSize(m_hDevices[m_nOperateID],&unPacketSize);
            GX_VERIFY(emStatus);

            emStatus = GXSetInt(m_hDevices[m_nOperateID], GX_INT_GEV_PACKETSIZE, unPacketSize);
            GX_VERIFY(emStatus);
        }
    }

	//Update the name of current operating device to the headline  
	SetWindowText(m_pBaseinfo[m_nOperateID].szDisplayName);
	GX_VERIFY(emStatus);
	m_pstCam[m_nOperateID].bIsOpen = TRUE;

	//Check whether the device supports PixelColorFilter
	emStatus = GXIsImplemented(m_hDevices[m_nOperateID], GX_ENUM_PIXEL_COLOR_FILTER, &bIsImplemented);
	GX_VERIFY(emStatus);
	m_pstCam[m_nOperateID].bIsColorFilter = bIsImplemented;
	if (bIsImplemented)
	{
		emStatus = GXGetEnum(m_hDevices[m_nOperateID], GX_ENUM_PIXEL_COLOR_FILTER, &nValue);
		GX_VERIFY(emStatus);
		m_pstCam[m_nOperateID].nBayerLayout = nValue;
	}

	// Get image width
	emStatus = GXGetInt(m_hDevices[m_nOperateID], GX_INT_WIDTH, &nValue);
	GX_VERIFY(emStatus);
	m_pstCam[m_nOperateID].nImageWidth = nValue;

	// Get image height
	emStatus = GXGetInt(m_hDevices[m_nOperateID], GX_INT_HEIGHT, &nValue);
	GX_VERIFY(emStatus);
	m_pstCam[m_nOperateID].nImageHeight = nValue;


	// If the device supports the pixel format of"GX_PIXEL_FORMAT_BAYER_GR8",
	// the following code can be used to set the pixel format to GX_PIXEL_FORMAT_BAYER_GR8:
	// emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_GR8);
	// VERIFY_STATUS_RET(emStatus);

	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8bit(m_hDevices[m_nOperateID]); 
	GX_VERIFY(emStatus);

	// Get the image payload size 
	emStatus = GXGetInt(m_hDevices[m_nOperateID], GX_INT_PAYLOAD_SIZE, &nValue);
	GX_VERIFY(emStatus);
	m_pstCam[m_nOperateID].nPayLoadSise = nValue;

	//Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevices[m_nOperateID], GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	GX_VERIFY(emStatus);

	//Set the TriggerMode to OFF 
	emStatus = GXSetEnum(m_hDevices[m_nOperateID], GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
	GX_VERIFY(emStatus);

	m_objMenu.EnableMenuItem(1, MF_BYPOSITION | MF_ENABLED);

	SetMenu(&m_objMenu);
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\return  emStatus GX_STATUS_SUCCESS: Success,else:Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CMainFrame::SetPixelFormat8bit(GX_DEV_HANDLE hDevice)
{
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	int64_t   nPixelSize  = 0;
	uint32_t  nEnmuEntry  = 0;
	size_t    nBufferSize = 0;
	BOOL      bIs8bit     = TRUE;

	GX_ENUM_DESCRIPTION  *pEnumDescription = NULL;
	GX_ENUM_DESCRIPTION  *pEnumTemp        = NULL;

	// Get the feature PixelSize, this feature indicates the depth of the pixel values in the acquired images in bits per pixel
	emStatus = GXGetEnum(hDevice, GX_ENUM_PIXEL_SIZE, &nPixelSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		return emStatus;
	}

	// If the PixelSize is 8bit then return, or set the PixelSize to 8bit.
	if (nPixelSize == GX_PIXEL_SIZE_BPP8)
	{
		return GX_STATUS_SUCCESS;
	}
	else
	{
		// Get the enumeration entry of the pixel format the device supports.
		emStatus = GXGetEnumEntryNums(hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			return emStatus;
		}

		// Allocate memory for getting the enumeration entry of the pixel format.
		nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
		pEnumDescription = new GX_ENUM_DESCRIPTION[nEnmuEntry];

		emStatus = GXGetEnumDescription(hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			if (pEnumDescription != NULL)
			{
				delete []pEnumDescription;
				pEnumDescription = NULL;
			}
			return emStatus;
		}

		// A loop to visit every enumeration entry node once to find which enumeration entry node is 8bit and then set pixel format to its value 
		for (uint32_t i = 0; i < nEnmuEntry; i++)
		{
			if ((pEnumDescription[i].nValue & GX_PIXEL_8BIT) == GX_PIXEL_8BIT)
			{
				emStatus = GXSetEnum(hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription[i].nValue);
				break;
			}
		}	

		// Release memory
		if (pEnumDescription != NULL)
		{
			delete []pEnumDescription;
			pEnumDescription = NULL;
		}
	}

	return emStatus;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close current device

\return  void
*/
//---------------------------------------------------------------------------------
void CMainFrame::OnMenuClose() 
{
	// TODO: Add your command handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CMenu*    pMenuSub = NULL;

	if (m_pstCam[m_nOperateID].bIsSnap)
	{
		// If the device is acquiring image then stop it.
		emStatus = GXSendCommand(m_hDevices[m_nOperateID], GX_COMMAND_ACQUISITION_STOP);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}

		//Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevices[m_nOperateID]);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}

		// Release memory for showing image
		if(m_pstCam[m_nOperateID].pImageBuffer != NULL)
		{
			delete[]m_pstCam[m_nOperateID].pImageBuffer;
			m_pstCam[m_nOperateID].pImageBuffer = NULL;
		}

		if (m_pstCam[m_nOperateID].pRawBuffer != NULL)
		{
			delete []m_pstCam[m_nOperateID].pRawBuffer;
			m_pstCam[m_nOperateID].pRawBuffer = NULL;
		}

		::ReleaseDC(m_pCWnds[m_nOperateID]->m_hWnd,m_hDCs[m_nOperateID]);
		m_hDCs[m_nOperateID] = NULL;

		m_pstCam[m_nOperateID].bIsSnap = FALSE;
	}

	emStatus = GXCloseDevice(m_hDevices[m_nOperateID]);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}
	m_hDevices[m_nOperateID]       = NULL;

	m_pstCam[m_nOperateID].bIsOpen = FALSE;
	pMenuSub = m_objMenu.GetSubMenu(0);      
	m_objMenu.EnableMenuItem(1, MF_BYPOSITION | MF_DISABLED | MF_GRAYED);
	SetMenu(&m_objMenu);
}

//---------------------------------------------------------------------------------
/**
\brief	This function demonstrates how to register multiple CaptureCallback functions
\param	CamID   [in]  

\return void
*/
//---------------------------------------------------------------------------------
void CMainFrame::RegisterCallback(int nCamID)
{
	GX_STATUS emStatus = GX_STATUS_ERROR;

	//Register the CaptureCallback function 
	switch(nCamID)
	{
	case 0:
		emStatus = GXRegisterCaptureCallback(m_hDevices[nCamID], this, OnFrameCallbackFun1);
		GX_VERIFY(emStatus);
		break;

	case 1:
		emStatus = GXRegisterCaptureCallback(m_hDevices[nCamID], this, OnFrameCallbackFun2);
		GX_VERIFY(emStatus);
		break;

	case 2:
		emStatus = GXRegisterCaptureCallback(m_hDevices[nCamID], this, OnFrameCallbackFun3);
		GX_VERIFY(emStatus);
		break;

	case 3:
		emStatus = GXRegisterCaptureCallback(m_hDevices[nCamID], this, OnFrameCallbackFun4);
		GX_VERIFY(emStatus);
		break;

	default:
		break;
	}

}

//---------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing color image.
\return void
*/
//---------------------------------------------------------------------------------
void CMainFrame::OnMenuStart() 
{
	// TODO: Add your command handler code here
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	bool      bIsColorFilter = m_pstCam[m_nOperateID].bIsColorFilter;

	// Allocate DC for showing image 
	if (m_hDCs[m_nOperateID] == NULL)
	{
		m_hDCs[m_nOperateID]   = ::GetDC(m_pCWnds[m_nOperateID]->m_hWnd);
	}

	// Initialize bitmap header 
	m_pstCam[m_nOperateID].pBmpInfo		                        = (BITMAPINFO *)(m_pstCam[m_nOperateID].chBmpBuf);
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biSize           = sizeof(BITMAPINFOHEADER);
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biWidth			= (LONG)m_pstCam[m_nOperateID].nImageWidth;
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biHeight			= (LONG)m_pstCam[m_nOperateID].nImageHeight;

	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biPlanes			= 1;
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biBitCount		= bIsColorFilter ? 24 : 8; 
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biCompression	= BI_RGB;
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biSizeImage		= 0;
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biXPelsPerMeter	= 0;
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biYPelsPerMeter	= 0;
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biClrUsed		= 0;
	m_pstCam[m_nOperateID].pBmpInfo->bmiHeader.biClrImportant	= 0;

	// If using mono cameras,you need to initialize color palette first.
	if (!bIsColorFilter)
	{
		for(int i = 0; i < 256; i++)
		{
			m_pstCam[m_nOperateID].pBmpInfo->bmiColors[i].rgbBlue	  = i;
			m_pstCam[m_nOperateID].pBmpInfo->bmiColors[i].rgbGreen	  = i;
			m_pstCam[m_nOperateID].pBmpInfo->bmiColors[i].rgbRed	  = i;
			m_pstCam[m_nOperateID].pBmpInfo->bmiColors[i].rgbReserved = i;
		}
	}
	
	// Release memory
	if(m_pstCam[m_nOperateID].pImageBuffer != NULL)
	{
		delete[]m_pstCam[m_nOperateID].pImageBuffer;
		m_pstCam[m_nOperateID].pImageBuffer = NULL;
	}

	if (m_pstCam[m_nOperateID].pRawBuffer != NULL)
	{
		delete []m_pstCam[m_nOperateID].pRawBuffer;
		m_pstCam[m_nOperateID].pRawBuffer = NULL;
	}

	// Allocate memory
	m_pstCam[m_nOperateID].pRawBuffer = new BYTE[(size_t)(m_pstCam[m_nOperateID].nPayLoadSise)];
	if(m_pstCam[m_nOperateID].pRawBuffer == NULL)
	{
		MessageBox("Fail to allocate memory!");
		return;
	}

	// Allocate memory if the device supports ColorFilter
	if(bIsColorFilter)
	{
		m_pstCam[m_nOperateID].pImageBuffer = new BYTE[(size_t)(m_pstCam[m_nOperateID].nImageWidth * m_pstCam[m_nOperateID].nImageHeight * 3)];
		if (m_pstCam[m_nOperateID].pImageBuffer == NULL)
		{
			MessageBox("Fail to allocate memory!");
			delete []m_pstCam[m_nOperateID].pRawBuffer;
			m_pstCam[m_nOperateID].pRawBuffer = NULL;
			return;
		}
	}

	// Allocate memory if the device is mono camera
	else
	{
		m_pstCam[m_nOperateID].pImageBuffer = new BYTE[(size_t)(m_pstCam[m_nOperateID].nImageWidth * m_pstCam[m_nOperateID].nImageHeight)];
		if (m_pstCam[m_nOperateID].pImageBuffer == NULL)
		{
			MessageBox("Fail to allocate memory!");
			delete []m_pstCam[m_nOperateID].pRawBuffer;
			m_pstCam[m_nOperateID].pRawBuffer = NULL;
			return;
		}
	}

	RegisterCallback(m_nOperateID);	

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevices[m_nOperateID], GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	GX_VERIFY(emStatus);

	//Send AquisitionStart command
	emStatus = GXSendCommand(m_hDevices[m_nOperateID], GX_COMMAND_ACQUISITION_START);
	GX_VERIFY(emStatus);

	m_pstCam[m_nOperateID].bIsSnap = TRUE;
}

//---------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to send "AcquisitionStop" command 

\return void
*/
//---------------------------------------------------------------------------------
void CMainFrame::OnMenuStop() 
{
	// TODO: Add your command handler code here
	GX_STATUS emStatus = GX_STATUS_ERROR;

	//Send AcquisitionStop command 
	emStatus = GXSendCommand(m_hDevices[m_nOperateID], GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);

	//Unregister the CaptureCallback function
	emStatus = GXUnregisterCaptureCallback(m_hDevices[m_nOperateID]);
	GX_VERIFY(emStatus);

	//Release memory for showing image
	if(m_pstCam[m_nOperateID].pImageBuffer != NULL)
	{
		delete[]m_pstCam[m_nOperateID].pImageBuffer;
		m_pstCam[m_nOperateID].pImageBuffer = NULL;
	}

	if (m_pstCam[m_nOperateID].pRawBuffer != NULL)
	{
		delete []m_pstCam[m_nOperateID].pRawBuffer;
		m_pstCam[m_nOperateID].pRawBuffer = NULL;
	}

	m_pCamsFps[m_nOperateID].Reset();

	::ReleaseDC(m_pCWnds[m_nOperateID]->m_hWnd,m_hDCs[m_nOperateID]);
	m_hDCs[m_nOperateID] = NULL;

	m_pstCam[m_nOperateID].bIsSnap = FALSE;
}

//---------------------------------------------------------------------------------
/**
\brief  This is the CaptureCallback function for first camera
\param  pFrame  

\return void
*/
//---------------------------------------------------------------------------------
void __stdcall CMainFrame::OnFrameCallbackFun1(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	if (pFrame->status != 0)
	{
		return;
	}

	CMainFrame  *pf = (CMainFrame*)(pFrame->pUserParam);
	int		    nID = 0;            

	BYTE	    *pImageBuffer = NULL;          //The pointer for the image has been processed 
	BYTE        *pRawBuf      = NULL;          //The pointer for the raw data of image
	BITMAPINFO	*pBmpInfo     = NULL;          //The buffer for BMP image
	int64_t	    nBayerLayout  = 0;             //PixelColorFilter
	int		    nImgWidth     = 0;             //ImgWidth
	int         nImgHeight    = 0;             //ImgHeight
	int        	nWndWidth     = 0;             //WindowWidth
	int  	    nWndHeight    = 0;             //WindowHeight
	int         i             = 0;             //The loop variable
	CRect       objRect;                       //The size of window
	HDC         objMemDC;                      //The memory DC
	HBITMAP     objMemBmp;                     //The bitmap that the device supports 
	char        chPutStr[MAX_PATH]  = {'\0'};  //The string for showing fps

	pf->m_pCamsFps[nID].IncreaseFrameNum();
	nImgWidth    = (int)(pf->m_pstCam[nID].nImageWidth);
	nImgHeight   = (int)(pf->m_pstCam[nID].nImageHeight);
	pImageBuffer = pf->m_pstCam[nID].pImageBuffer;
	pRawBuf      = pf->m_pstCam[nID].pRawBuffer;
	nBayerLayout = pf->m_pstCam[nID].nBayerLayout;
	pBmpInfo     = pf->m_pstCam[nID].pBmpInfo;
	pf->m_pCWnds[0]->GetClientRect(objRect); 
	nWndWidth    = objRect.Width();
	nWndHeight   = objRect.Height();
	
	//Copy the image to the RawBuffer
	memcpy(pf->m_pstCam[nID].pRawBuffer, pFrame->pImgBuf, (size_t)(pf->m_pstCam[nID].nPayLoadSise));

	
	if(pf->m_pstCam[nID].bIsColorFilter)   
	{
		 //Show image after converting image data if color camera has been used.
		DxRaw8toRGB24(pRawBuf, pImageBuffer, nImgWidth, nImgHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayerLayout), TRUE);
	}
	else     
	{
		//Show image after flipping image data if mono camera has been used.
		for(i = 0; i < nImgHeight; i++)
		{
			memcpy((pImageBuffer + i * nImgWidth), (pRawBuf + ( nImgHeight - i - 1) * nImgWidth), nImgWidth);
		}
	}

	double          dTime         = 0;       
	double          dSum          = 0;       

	if (!pf->m_bViewID)       // Show image without showing fps
	{
		//You could use the following code to show image.
		::SetStretchBltMode(pf->m_hDCs[0], COLORONCOLOR);
		StretchDIBits(pf->m_hDCs[0],
			0,						
			0,
			nWndWidth,
			nWndHeight,
			0,
			0,
			nImgWidth,
			nImgHeight,
			pImageBuffer,
			pBmpInfo,
			DIB_RGB_COLORS,
		    SRCCOPY);
	}
	else           // Show image and fps
	{
		objMemDC = ::CreateCompatibleDC(pf->m_hDCs[0]);
		objMemBmp= CreateCompatibleBitmap(pf->m_hDCs[0], nWndWidth, nWndHeight);
		::SelectObject(objMemDC,objMemBmp);

		// You must use the following code to show image.
		::SetStretchBltMode(objMemDC, COLORONCOLOR);
		StretchDIBits(objMemDC,
			0,						
			0,
		    nWndWidth,
		    nWndHeight,
		    0,
		    0,
		    nImgWidth,
		    nImgHeight,
		    pImageBuffer,
		    pBmpInfo,
		    DIB_RGB_COLORS,
		    SRCCOPY);

		//Show the fps and the serial number 
		sprintf(chPutStr, "Serial number:1 SN:%s FPS:%.2f", pf->m_pBaseinfo[nID].szSN, pf->m_pstCam[nID].fFps);
		TextOut(objMemDC,0, 0, chPutStr, (int)strlen(chPutStr));
		StretchBlt(pf->m_hDCs[0],
			0,
	    	0,
		    nWndWidth,
		    nWndHeight,
		    objMemDC,
		    0,
		    0,
		    nWndWidth,
		    nWndHeight,
		    SRCCOPY);

		::DeleteDC(objMemDC);
		DeleteObject(objMemBmp);
	}
}

//---------------------------------------------------------------------------------
/**
\brief This is the CaptureCallback function for second camera
\param pFrame 
\return void
*/
//---------------------------------------------------------------------------------
void __stdcall CMainFrame::OnFrameCallbackFun2(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	if (pFrame->status != 0)
	{
		return;
	}

	CMainFrame *pf = (CMainFrame*)(pFrame->pUserParam);
	int		   nID = 1;            

	BYTE	    *pImageBuffer = NULL;          //The pointer for the image has been processed 
	BYTE        *pRawBuf      = NULL;          //The pointer for the raw data of image
	BITMAPINFO	*pBmpInfo     = NULL;          //The buffer for BMP image
	int64_t	    nBayerLayout  = 0;             //PixelColorFilter
	int		    nImgWidth     = 0;             //ImgWidth
	int         nImgHeight    = 0;             //ImgHeight
	int        	nWndWidth     = 0;             //WindowWidth
	int  	    nWndHeight    = 0;             //WindowHeight
	int         i             = 0;             //The loop variable
	CRect       objRect;                       //The size of window
	HDC         objMemDC;                      //The memory DC
	HBITMAP     objMemBmp;                     //The bitmap that the device supports 
	char        chPutStr[MAX_PATH]  = {'\0'};        //The string for showing fps
	
	pf->m_pCamsFps[nID].IncreaseFrameNum();
	nImgWidth    = (int)(pf->m_pstCam[nID].nImageWidth);
	nImgHeight   = (int)(pf->m_pstCam[nID].nImageHeight);
	pImageBuffer = pf->m_pstCam[nID].pImageBuffer;
	pRawBuf      = pf->m_pstCam[nID].pRawBuffer;
	nBayerLayout = pf->m_pstCam[nID].nBayerLayout;
	pBmpInfo     = pf->m_pstCam[nID].pBmpInfo;
	pf->m_pCWnds[1]->GetClientRect(objRect); 
	nWndWidth    = objRect.Width();
	nWndHeight   = objRect.Height();

	//Copy the image to the RawBuffer
	memcpy(pf->m_pstCam[nID].pRawBuffer,pFrame->pImgBuf,(size_t)(pf->m_pstCam[nID].nPayLoadSise));

	if(pf->m_pstCam[nID].bIsColorFilter)  
	{  
		//Show image after converting image data if color camera has been used.
		DxRaw8toRGB24(pRawBuf, pImageBuffer, nImgWidth, nImgHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayerLayout), TRUE);
	}
	else   
	{
		//Show image after flipping image data if mono camera has been used.
		for(i = 0; i < nImgHeight; i++)
		{
			memcpy(pImageBuffer + i * nImgWidth, pRawBuf + (nImgHeight - i - 1) * nImgWidth, nImgWidth);
		}
	}

	if (!pf->m_bViewID)       // Show image without showing fps
	{
		//You must use the following code to show image.
		::SetStretchBltMode(pf->m_hDCs[1], COLORONCOLOR);
		StretchDIBits(pf->m_hDCs[1],
			0,						
			0,
			nWndWidth,
			nWndHeight,
			0,
			0,
			nImgWidth,
			nImgHeight,
			pImageBuffer,
			pBmpInfo,
			DIB_RGB_COLORS,
			SRCCOPY);
	}
	else           // Show image and fps
	{
		objMemDC = ::CreateCompatibleDC(pf->m_hDCs[1]);
		objMemBmp= CreateCompatibleBitmap(pf->m_hDCs[1], nWndWidth, nWndHeight);
		::SelectObject(objMemDC,objMemBmp);
		
		// You must use the following code to show image.
		::SetStretchBltMode(objMemDC, COLORONCOLOR);
		StretchDIBits(objMemDC,
			0,						
			0,
			nWndWidth,
			nWndHeight,
			0,
			0,
			nImgWidth,
			nImgHeight,
			pImageBuffer,
			pBmpInfo,
			DIB_RGB_COLORS,
			SRCCOPY);
		
		//Show the fps and the serial number 
		sprintf(chPutStr, "Serial number:2 SN:%s FPS:%.2f", pf->m_pBaseinfo[nID].szSN, pf->m_pstCam[nID].fFps);
		TextOut(objMemDC,0, 0, chPutStr, (int)strlen(chPutStr));
		StretchBlt(pf->m_hDCs[1],
			0,
			0,
			nWndWidth,
			nWndHeight,
			objMemDC,
			0,
			0,
			nWndWidth,
			nWndHeight,
			SRCCOPY);
		::DeleteDC(objMemDC);
		DeleteObject(objMemBmp);
	}
}

//---------------------------------------------------------------------------------
/**
\brief This is the CaptureCallback function for third camera
\param pFrame  

\return void
*/
//---------------------------------------------------------------------------------
void __stdcall CMainFrame::OnFrameCallbackFun3(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	if (pFrame->status != 0)
	{
		return;
	}
	CMainFrame *pf = (CMainFrame*)(pFrame->pUserParam);
	int		   nID = 2;             

	BYTE	    *pImageBuffer = NULL;          //The pointer for the image has been processed 
	BYTE        *pRawBuf      = NULL;          //The pointer for the raw data of image
	BITMAPINFO	*pBmpInfo     = NULL;          //The buffer for BMP image
	int64_t	    nBayerLayout  = 0;             //PixelColorFilter
	int		    nImgWidth     = 0;             //ImgWidth
	int         nImgHeight    = 0;             //ImgHeight
	int        	nWndWidth     = 0;             //WindowWidth
	int  	    nWndHeight    = 0;             //WindowHeight
	int         i             = 0;             //The loop variable
	CRect       objRect;                       //The size of window
	HDC         objMemDC;                      //The memory DC
	HBITMAP     objMemBmp;                     //The bitmap that the device supports 
	char        chPutStr[MAX_PATH]  = {'\0'};  //The string for showing fps
	

	pf->m_pCamsFps[nID].IncreaseFrameNum();
	nImgWidth    = (int)(pf->m_pstCam[nID].nImageWidth);
	nImgHeight   = (int)(pf->m_pstCam[nID].nImageHeight);
	pImageBuffer = pf->m_pstCam[nID].pImageBuffer;
	pRawBuf      = pf->m_pstCam[nID].pRawBuffer;
	nBayerLayout = pf->m_pstCam[nID].nBayerLayout;
	pBmpInfo     = pf->m_pstCam[nID].pBmpInfo;
	pf->m_pCWnds[2]->GetClientRect(objRect); 
	nWndWidth    = objRect.Width();
	nWndHeight   = objRect.Height();

	//Copy the image to the RawBuffer
	memcpy(pf->m_pstCam[nID].pRawBuffer, pFrame->pImgBuf, (size_t)(pf->m_pstCam[nID].nPayLoadSise));
	
	
	if(pf->m_pstCam[nID].bIsColorFilter)   
	{   
		//Show image after converting image data if color camera has been used.
		DxRaw8toRGB24(pRawBuf, pImageBuffer, nImgWidth, nImgHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayerLayout), TRUE);
	}
	else    
	{
		//Show image after flipping image data if mono camera has been used.
		for(i = 0; i < nImgHeight; i++)
		{
			memcpy(pImageBuffer + i * nImgWidth, pRawBuf + (nImgHeight - i - 1) * nImgWidth, nImgWidth);
		}
	}

	if (!pf->m_bViewID)       // Show image without showing fps
	{
		//You must use the following code to show image.
		::SetStretchBltMode(pf->m_hDCs[2], COLORONCOLOR);
		StretchDIBits(pf->m_hDCs[2],
			0,						
			0,
			nWndWidth,
			nWndHeight,
			0,
			0,
			nImgWidth,
			nImgHeight,
			pImageBuffer,
			pBmpInfo,
			DIB_RGB_COLORS,
			SRCCOPY);
	}
	else           // Show image and fps
	{
		objMemDC = ::CreateCompatibleDC(pf->m_hDCs[2]);
		objMemBmp= CreateCompatibleBitmap(pf->m_hDCs[2], nWndWidth, nWndHeight);
		::SelectObject(objMemDC,objMemBmp);

		// You must use the following code to show image.
		::SetStretchBltMode(objMemDC, COLORONCOLOR);
		StretchDIBits(objMemDC,
			0,						
			0,
			nWndWidth,
			nWndHeight,
			0,
			0,
			nImgWidth,
			nImgHeight,
			pImageBuffer,
			pBmpInfo,
			DIB_RGB_COLORS,
			SRCCOPY);
		
		//Show the fps and the serial number 
		sprintf(chPutStr, "Serial number:3 SN:%s FPS:%.2f", pf->m_pBaseinfo[nID].szSN, pf->m_pstCam[nID].fFps);
		TextOut(objMemDC,0, 0, chPutStr, (int)strlen(chPutStr));
		StretchBlt(pf->m_hDCs[2],
			0,
			0,
			nWndWidth,
			nWndHeight,
			objMemDC,
			0,
			0,
			nWndWidth,
			nWndHeight,
			SRCCOPY);
		::DeleteDC(objMemDC);
		DeleteObject(objMemBmp);
	}
}

//---------------------------------------------------------------------------------
/**
\brief  This is the CaptureCallback function for forth camera
\param  pFrame 

\return void
*/
//---------------------------------------------------------------------------------
void __stdcall CMainFrame::OnFrameCallbackFun4(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	if (pFrame->status != 0)
	{
		return;
	}

	CMainFrame *pf = (CMainFrame*)(pFrame->pUserParam);
	int		   nID = 3;            

	BYTE	    *pImageBuffer = NULL;          //The pointer for the image has been processed 
	BYTE        *pRawBuf      = NULL;          //The pointer for the raw data of image
	BITMAPINFO	*pBmpInfo     = NULL;          //The buffer for BMP image
	int64_t	    nBayerLayout  = 0;             //PixelColorFilter
	int		    nImgWidth     = 0;             //ImgWidth
	int         nImgHeight    = 0;             //ImgHeight
	int        	nWndWidth     = 0;             //WindowWidth
	int  	    nWndHeight    = 0;             //WindowHeight
	int         i             = 0;             //The loop variable
	CRect       objRect;                       //The size of window
	HDC         objMemDC;                      //The memory DC
	HBITMAP     objMemBmp;                     //The bitmap that the device supports 
	char        chPutStr[MAX_PATH]  = {'\0'};  //The string for showing fps
	
	
	pf->m_pCamsFps[nID].IncreaseFrameNum();
	nImgWidth    = (int)(pf->m_pstCam[nID].nImageWidth);
	nImgHeight   = (int)(pf->m_pstCam[nID].nImageHeight);
	pImageBuffer = pf->m_pstCam[nID].pImageBuffer;
	pRawBuf      = pf->m_pstCam[nID].pRawBuffer;
	nBayerLayout = pf->m_pstCam[nID].nBayerLayout;
	pBmpInfo     = pf->m_pstCam[nID].pBmpInfo;
	pf->m_pCWnds[3]->GetClientRect(objRect); 
	nWndWidth    = objRect.Width();
	nWndHeight   = objRect.Height();

	//Copy the image to the RawBuffer
	memcpy(pf->m_pstCam[nID].pRawBuffer, pFrame->pImgBuf, (size_t)(pf->m_pstCam[nID].nPayLoadSise));

	if(pf->m_pstCam[nID].bIsColorFilter) 
	{ 
		//Show image after converting image data if color camera has been used.
		DxRaw8toRGB24(pRawBuf, pImageBuffer, nImgWidth, nImgHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayerLayout), TRUE);
	}
	else         
	{
		//Show image after flipping image data if mono camera has been used.
		for(i = 0; i < nImgHeight; i++)
		{
			memcpy(pImageBuffer + i * nImgWidth, pRawBuf + (nImgHeight - i - 1) * nImgWidth, nImgWidth);
		}
	}

	if (!pf->m_bViewID)       // Show image without showing fps
	{
		//You must use the following code to show image.
		int nRet = ::SetStretchBltMode(pf->m_hDCs[3], COLORONCOLOR);
		int n2   = StretchDIBits(pf->m_hDCs[3],
			0,						
			0,
			nWndWidth,
			nWndHeight,
			0,
			0,
			nImgWidth,
			nImgHeight,
			pImageBuffer,
			pBmpInfo,
			DIB_RGB_COLORS,
			SRCCOPY);
	}
	else           // Show image and fps
	{
		objMemDC = ::CreateCompatibleDC(pf->m_hDCs[3]);
		objMemBmp= CreateCompatibleBitmap(pf->m_hDCs[3], nWndWidth, nWndHeight);
		::SelectObject(objMemDC,objMemBmp);

		// You must use the following code to show image.
		::SetStretchBltMode(objMemDC, COLORONCOLOR);
		StretchDIBits(objMemDC,
			0,						
			0,
			nWndWidth,
			nWndHeight,
			0,
			0,
			nImgWidth,
			nImgHeight,
			pImageBuffer,
			pBmpInfo,
			DIB_RGB_COLORS,
			SRCCOPY);
		
		//Show the fps and the serial number 
		sprintf(chPutStr, "Serial number:4 SN:%s FPS:%.2f", pf->m_pBaseinfo[nID].szSN, pf->m_pstCam[nID].fFps);
		TextOut(objMemDC,0, 0, chPutStr, (int)strlen(chPutStr));
		StretchBlt(pf->m_hDCs[3],
			0,
			0,
			nWndWidth,
			nWndHeight,
			objMemDC,
			0,
			0,
			nWndWidth,
			nWndHeight,
			SRCCOPY);
		::DeleteDC(objMemDC);
		DeleteObject(objMemBmp);
	}
}

void CMainFrame::OnSize(UINT nType, int cx, int cy) 
{
	CFrameWnd::OnSize(nType, cx, cy);	
}

void CMainFrame::OnUpdateMenuOpen(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if((m_nCamNum > 0) && (!m_pstCam[m_nOperateID].bIsOpen))
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}
}

void CMainFrame::OnUpdateMenuStart(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if((m_nCamNum > 0) && (m_pstCam[m_nOperateID].bIsOpen) && (!m_pstCam[m_nOperateID].bIsSnap))
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}
}

void CMainFrame::OnUpdateMenuClose(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if((m_nCamNum > 0) && (m_pstCam[m_nOperateID].bIsOpen))
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}
}

void CMainFrame::OnUpdateMenuStop(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	if((m_nCamNum > 0) && (m_pstCam[m_nOperateID].bIsOpen) && (m_pstCam[m_nOperateID].bIsSnap))
	{
		pCmdUI->Enable(TRUE);
	}
	else
	{
		pCmdUI->Enable(FALSE);
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus   [in] 

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::ShowErrorString(GX_STATUS emErrorStatus)
{
	char*     pchErrorInfo = NULL;
	size_t    nSize        = 0;
	GX_STATUS emStatus     = GX_STATUS_ERROR;
	
	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (NULL == pchErrorInfo)
	{
		return;
	}

	// Get error information and show them to user 
	emStatus = GXGetLastError (&emErrorStatus, pchErrorInfo, &nSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		MessageBox("Fail to call GXGetLastError interface!");
	}
	else
	{
		MessageBox((LPCTSTR)pchErrorInfo);
	}

	// Release memory 
	if (NULL != pchErrorInfo)
	{
		delete[] pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

void CMainFrame::OnMenuCamid() 
{
	// TODO: Add your command handler code here
	m_bViewID = !m_bViewID;
}

void CMainFrame::OnUpdateMenuCamid(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->SetCheck(m_bViewID);
}

//----------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to compute the fps
\param    nIDEvent 

\return   void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnTimer(UINT_PTR nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default
	uint32_t i = 0;

	if((m_nCamNum <= 0) || (m_nOperateID < 0))
	{
		CFrameWnd::OnTimer(nIDEvent);
		return;
	}

	// Update the frame rate
	for(i = 0; i < m_nCamNum; i++)
	{
		m_pCamsFps[i].UpdateFps();
		m_pstCam[i].fFps = (float)m_pCamsFps[i].GetFps();
	}
	
	CFrameWnd::OnTimer(nIDEvent);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close the application.

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_ERROR;

	
	KillTimer(0);

	
	for(uint32_t i = 0; i < m_nCamNum; i++)
	{
		if(m_pstCam[i].bIsSnap)
		{
			//Send AcquisitionStop command
			emStatus = GXSendCommand(m_hDevices[i], GX_COMMAND_ACQUISITION_STOP);

			//Unregister the CaptureCallback function 
			emStatus = GXUnregisterCaptureCallback(m_hDevices[i]);
		}
		if(m_pstCam[i].bIsOpen)
		{
			emStatus      = GXCloseDevice(m_hDevices[i]);
			m_hDevices[i] = NULL;
		}
	}
	CFrameWnd::OnClose();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to create 4 windows

\return void
*/
//----------------------------------------------------------------------------------
void CMainFrame::OnMenuCammerParam() 
{
	// TODO: Add your command handler code here
	CCamerParamDlg dlg;
	dlg.DoModal();	
}

BOOL CMainFrame::OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext) 
{
	// TODO: Add your specialized code here and/or call the base class
	CRect  objRect;
	GetClientRect(&objRect);
	
	//Split the static control to 4 windows
	CSize paneSize(objRect.Width() / 2, objRect.Height() / 2);
	m_objSplitter.CreateStatic(this, 2, 2);

	//Create the first window
	m_objSplitter.CreateView(0, 0, RUNTIME_CLASS(CGxMultiCamView), paneSize, pContext);
    m_pCWnds[0]	= m_objSplitter.GetPane(0, 0);

	//Create the second window
	m_objSplitter.CreateView(0, 1, RUNTIME_CLASS(CGxMultiCamView), paneSize, pContext);
	m_pCWnds[1] = m_objSplitter.GetPane(0, 1);

	//Create the third window
	m_objSplitter.CreateView(1, 0, RUNTIME_CLASS(CGxMultiCamView), paneSize, pContext);
	m_pCWnds[2]	= m_objSplitter.GetPane(1, 0);

	//Create the fourth window
	m_objSplitter.CreateView(1, 1, RUNTIME_CLASS(CGxMultiCamView), paneSize, pContext);
	m_pCWnds[3] = m_objSplitter.GetPane(1, 1);

	return TRUE;
}

