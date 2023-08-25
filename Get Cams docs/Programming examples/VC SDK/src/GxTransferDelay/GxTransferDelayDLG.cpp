// GxTransferDelayDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxTransferDelay.h"
#include "GxTransferDelayDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	//{{AFX_DATA(CAboutDlg)
	enum { IDD = IDD_ABOUTBOX };
	CString	m_strProductVersion;
	CString	m_strLegalCopyRight;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CAboutDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	//{{AFX_MSG(CAboutDlg)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
	//{{AFX_DATA_INIT(CAboutDlg)
	m_strProductVersion = _T("");
	m_strLegalCopyRight = _T("");
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	DDX_Text(pDX, IDC_STATIC_PRODUCTVERSION, m_strProductVersion);
	DDX_Text(pDX, IDC_STATIC_LEGALCOPYRIGHT, m_strLegalCopyRight);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxTransferDelayDlg dialog

CGxTransferDelayDlg::CGxTransferDelayDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxTransferDelayDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CGxTransferDelayDlg)
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_nDevNum                     = 0;            // The number of connected devices 
	m_pDeviceHandle               = NULL;		  // The pointer for handle
	m_pCBHandle                   = NULL;         // The pointer for the exposure callback function
	m_pstCamInfo                  = NULL;	      // The pointer for the camera data structure 
	m_bIsOpen                     = false;        // The flag indicates whether the device has been opened
	m_bIsSnap                     = false;        // The flag indicates whether image acquisition has started
	m_hCam0StartTranEvent         = NULL;         // The handle for Camera No.0 StartTransferEvent 
	m_hCam1StartTranEvent         = NULL;         // The handle for Camera No.1 StartTransferEvent 
	m_hCam2StartTranEvent         = NULL;         // The handle for Camera No.2 StartTransferEvent 
	m_hCam3StartTranEvent         = NULL;         // The handle for Camera No.3 StartTransferEvent 
	m_hCam0StopSnapEvent          = NULL;         // The handle for Camera No.0 AcquisitionStopEvent
	m_hCam1StopSnapEvent          = NULL;         // The handle for Camera No.1 AcquisitionStopEvent
	m_hCam2StopSnapEvent          = NULL;         // The handle for Camera No.2 AcquisitionStopEvent
	m_hCam3StopSnapEvent          = NULL;         // The handle for Camera No.3 AcquisitionStopEvent
	m_pLogWnd                     = NULL;         // The pointer of the log window 

	for (int i = 0; i < MAX_NUM_CAMERA; i++)
	{
		m_hWndHandle[i] = NULL;
		m_hDC[i]        = NULL;
	} 

}     

void CGxTransferDelayDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxTransferDelayDlg)
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxTransferDelayDlg, CDialog)
	//{{AFX_MSG_MAP(CGxTransferDelayDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_OPEN_DEVICE, OnOpenDevice)
	ON_BN_CLICKED(IDC_START_SNAP, OnStartSnap)
	ON_BN_CLICKED(IDC_STOP_SNAP, OnStopSnap)
	ON_BN_CLICKED(IDC_CLOSE_DEVICE, OnCloseDevice)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxTransferDelayDlg message handlers

BOOL CGxTransferDelayDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		CString strAboutMenu;
		strAboutMenu.LoadString(IDS_ABOUTBOX);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon
	
	// TODO: Add extra initialization here

	//Before using any GxIAPI methods, the GxIAPI must be initialized. 
	GX_STATUS emStatus = GX_STATUS_ERROR;
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}

	bool bReturn = InitResources();
	if (!bReturn)
	{
        MessageBox("Fail to initialize !");
		exit(0);
	}

	// The pointer of log window 
	m_pLogWnd = GetDlgItem(IDC_EDIT_LOG_WND);
	UpdateUI();

	// Get the pointer and handle for drawing image
	GetDlgItem(IDC_STATIC_SHOW_IMAGE_ZERO, &m_hWndHandle[0]);
	GetDlgItem(IDC_STATIC_SHOW_IMAGE_ONE, &m_hWndHandle[1]);
	GetDlgItem(IDC_STATIC_SHOW_IMAGE_TWO, &m_hWndHandle[2]);
	GetDlgItem(IDC_STATIC_SHOW_IMAGE_THREE, &m_hWndHandle[3]);
	for (int i = 0; i < MAX_NUM_CAMERA; i++)
	{
		m_hDC[i]  = ::GetDC(m_hWndHandle[i]);
		::GetClientRect(m_hWndHandle[i], &m_objRect[i]);	
	}  
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxTransferDelayDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char chFileName[MAX_PATH] = {'\0'};
        GetModuleFileName(NULL, chFileName, MAX_PATH);
        CFileVersion fv(chFileName);
        CAboutDlg dlgAbout;		
        dlgAbout.m_strProductVersion = _T("Version: ") + fv.GetProductVersion();
        dlgAbout.m_strLegalCopyRight = fv.GetLegalCopyright();
        dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CGxTransferDelayDlg::OnPaint() 
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CGxTransferDelayDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close the application.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	int i = 0;
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Stop acquiring images if the program has detected signal
		SetEvent(m_hCam0StopSnapEvent);
		SetEvent(m_hCam1StopSnapEvent);
		SetEvent(m_hCam2StopSnapEvent);
		SetEvent(m_hCam3StopSnapEvent);

		for (i = 0; i < m_nDevNum; i++)
		{
			// Send AcquisitionStop command
			emStatus = GXSendCommand(m_pDeviceHandle[i], GX_COMMAND_ACQUISITION_STOP);
			
			// Flush image queues to clear out-of-date images
			emStatus = GXFlushEvent(m_pDeviceHandle[i]);
		}	
	}

	// If the device has been opened then close it.
	if (m_bIsOpen)
	{
		for (i = 0; i < m_nDevNum; i++)
		{
			// Unregister the CaptureCallback function
			emStatus = GXUnregisterCaptureCallback(m_pDeviceHandle[i]);
			
			// Unregister the ExposureEvent callback function
			emStatus = GXUnregisterFeatureCallback(m_pDeviceHandle[i], GX_INT_EVENT_EXPOSUREEND, m_pCBHandle[i]);
		
			emStatus = GXCloseDevice(m_pDeviceHandle[i]);
			m_pDeviceHandle[i] = NULL;
		}
	}
	
	
	ReleaseResource();

	// Release GxIAPI
	emStatus = GXCloseLib();

	// Release the handle 
	for (i = 0; i < MAX_NUM_CAMERA; i++)
	{
		::ReleaseDC(m_hWndHandle[i], m_hDC[i]);
	}  
	
	CDialog::OnClose();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.

\return void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::OnOpenDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	uint32_t  nDeviceNum  = 0;
	char      chIndex[10] = {0};
	int       i           = 0;
	bool      bReturn     = false;

    GX_OPEN_PARAM stOpenParam;
    stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
    stOpenParam.openMode   = GX_OPEN_INDEX;
    stOpenParam.pszContent = "";

	// Enumerate all camera devices
	emStatus = GXUpdateDeviceList(&nDeviceNum, 1000);
	GX_VERIFY(emStatus);
	if (nDeviceNum <= 0)
	{
        MessageBox("No devices found!");
		return;
	}

	// This sample support 4 cameras at most
	m_nDevNum = min(nDeviceNum, MAX_NUM_CAMERA);

	for (i = 0; i < m_nDevNum; i++)
	{
		// The serial number begins with 1 
		_itoa(i + 1, chIndex, 10);

		// If the device is opened then close it to ensure the camera could open again.
		if (m_pDeviceHandle[i] != NULL)
		{
			emStatus = GXCloseDevice(m_pDeviceHandle[i]);
			GX_VERIFY(emStatus);
			m_pDeviceHandle[i] = NULL;
		}
		
		stOpenParam.pszContent = chIndex;
		//Open the first found device 
		emStatus = GXOpenDevice(&stOpenParam, &m_pDeviceHandle[i]);
		GX_VERIFY(emStatus);

        // It is suggested that users set the camera's packet size according to the current network environment after opening the network camera.
        // To improve the acquisition performance of network cameras, the following code is used for reference.
        {
            bool	 bImplementPacketSize = false;
            uint32_t unPacketSize		  = 0;

            emStatus = GXIsImplemented(m_pDeviceHandle[i], GX_INT_GEV_PACKETSIZE, &bImplementPacketSize);
            GX_VERIFY(emStatus);

            if (bImplementPacketSize)
            {
                emStatus = GXGetOptimalPacketSize(m_pDeviceHandle[i],&unPacketSize);
                GX_VERIFY(emStatus);

                emStatus = GXSetInt(m_pDeviceHandle[i], GX_INT_GEV_PACKETSIZE, unPacketSize);
                GX_VERIFY(emStatus);
            }
        }
		
		emStatus = SetCameraParam(m_pDeviceHandle[i]);
		GX_VERIFY(emStatus);
		
		emStatus = GetCameraParam(m_pDeviceHandle[i], &m_pstCamInfo[i]);
		GX_VERIFY(emStatus);
			
		bReturn = InitShowImageInfo(m_pDeviceHandle[i], &m_pstCamInfo[i]);
		if (!bReturn)
		{
            MessageBox("Fail to allocate resource for showing image!");
			return;
		}

		// Register the callback function
		emStatus = RegisterCallback(i);
		GX_VERIFY(emStatus);
	}	

	m_bIsOpen = true;
	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to register the CaptureCallback function                     
\param   nDevNum   
 
\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxTransferDelayDlg::RegisterCallback(int nDevNum)
{
	GX_STATUS emStatus = GX_STATUS_ERROR;

	// Register CaptureCallback function and ExposureEndEvent callback function 
 	switch(nDevNum)
 	{
 	case CAMERA0:

		emStatus = GXRegisterFeatureCallback(m_pDeviceHandle[CAMERA0], this, OnFeatureCallbackFun0, 
		                                     GX_INT_EVENT_EXPOSUREEND, &m_pCBHandle[CAMERA0]);
		VERIFY_STATUS_RET(emStatus); 
		
		emStatus = GXRegisterCaptureCallback(m_pDeviceHandle[CAMERA0], this, OnFrameCallbackFun0);
		VERIFY_STATUS_RET(emStatus);
		break;
		
	case CAMERA1:

		emStatus = GXRegisterFeatureCallback(m_pDeviceHandle[CAMERA1], this, OnFeatureCallbackFun1, 
			                                 GX_INT_EVENT_EXPOSUREEND, &m_pCBHandle[CAMERA1]);
		VERIFY_STATUS_RET(emStatus);
		
		emStatus = GXRegisterCaptureCallback(m_pDeviceHandle[CAMERA1], this, OnFrameCallbackFun1);
		VERIFY_STATUS_RET(emStatus);
		break;
		
	case CAMERA2:
		
		emStatus = GXRegisterFeatureCallback(m_pDeviceHandle[CAMERA2], this, OnFeatureCallbackFun2, 
		                           	         GX_INT_EVENT_EXPOSUREEND, &m_pCBHandle[CAMERA2]);
		VERIFY_STATUS_RET(emStatus);
		
		emStatus = GXRegisterCaptureCallback(m_pDeviceHandle[CAMERA2], this, OnFrameCallbackFun2);
		VERIFY_STATUS_RET(emStatus);
		break;
		
	case CAMERA3:

		emStatus = GXRegisterFeatureCallback(m_pDeviceHandle[CAMERA3], this, OnFeatureCallbackFun3, 
			                                 GX_INT_EVENT_EXPOSUREEND, &m_pCBHandle[CAMERA3]);
		VERIFY_STATUS_RET(emStatus);

		emStatus = GXRegisterCaptureCallback(m_pDeviceHandle[CAMERA3], this, OnFrameCallbackFun3);
		VERIFY_STATUS_RET(emStatus);
		break;
		
	default:
		break;
 	}
	return emStatus;
}
//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize devices 
\param   hDevice  

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxTransferDelayDlg::SetCameraParam(GX_DEV_HANDLE hDevice)
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8Bit(hDevice); 
	VERIFY_STATUS_RET(emStatus);

	// Set the continuous frame acquisition mode
	emStatus = GXSetEnum(hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);
	
	// Set the TriggerMode on
	emStatus = GXSetEnum(hDevice, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_ON);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerSelector
	emStatus = GXSetEnum(hDevice, GX_ENUM_TRIGGER_SELECTOR, GX_ENUM_TRIGGER_SELECTOR_FRAME_START);
	VERIFY_STATUS_RET(emStatus);
	
	// Select the TriggerSource
	emStatus = GXSetEnum(hDevice, GX_ENUM_TRIGGER_SOURCE, GX_TRIGGER_SOURCE_LINE0);
	VERIFY_STATUS_RET(emStatus);
	
	// Select the ExposureEndEvent
	emStatus = GXSetEnum(hDevice, GX_ENUM_EVENT_SELECTOR, GX_ENUM_EVENT_SELECTOR_EXPOSUREEND);
	VERIFY_STATUS_RET(emStatus);
	
	// Enable the ExposureEndEvent
	emStatus = GXSetEnum(hDevice, GX_ENUM_EVENT_NOTIFICATION, GX_ENUM_EVENT_NOTIFICATION_ON);
	VERIFY_STATUS_RET(emStatus);

	// Set the TransferContorlMode
	emStatus = GXSetEnum(hDevice, GX_ENUM_TRANSFER_CONTROL_MODE, GX_ENUM_TRANSFER_CONTROL_MODE_USERCONTROLED);
	VERIFY_STATUS_RET(emStatus);

	// Set the TransferOperationMode
	emStatus = GXSetEnum(hDevice, GX_ENUM_TRANSFER_OPERATION_MODE, GX_ENUM_TRANSFER_OPERATION_MODE_MULTIBLOCK);
	VERIFY_STATUS_RET(emStatus);

	// Set the TransferFrame count 
	emStatus = GXSetInt(hDevice, GX_INT_TRANSFER_BLOCK_COUNT, TRANSFER_COUNT);
	VERIFY_STATUS_RET(emStatus);

	return emStatus;
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\param   hDevice  

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxTransferDelayDlg::SetPixelFormat8Bit(GX_DEV_HANDLE hDevice)
{
	uint32_t  i           = 0;
	int64_t   nPixelSize  = 0;                     
	uint32_t  nEnmuEntry  = 0;
	size_t    nBufferSize = 0;	
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	GX_ENUM_DESCRIPTION  *pEnumDescription = NULL;
	GX_ENUM_DESCRIPTION  *pEnumTemp        = NULL;
	
	// Get the feature PixelSize, this feature indicates the depth of the pixel values in the acquired images in bits per pixel
	emStatus = GXGetEnum(hDevice, GX_ENUM_PIXEL_SIZE, &nPixelSize);
	VERIFY_STATUS_RET(emStatus);
	
	// If the PixelSize is 8bit then return
	if (nPixelSize == GX_PIXEL_SIZE_BPP8)
	{
		return GX_STATUS_SUCCESS;
	}
	
	// Get the enumeration entry of the pixel format the device supports.
	emStatus = GXGetEnumEntryNums(hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
	VERIFY_STATUS_RET(emStatus);
	
	// Allocate memory for getting the enumeration entry of the pixel format.
	nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
	pEnumDescription = new GX_ENUM_DESCRIPTION[nEnmuEntry];
	
	emStatus = GXGetEnumDescription(hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
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
			emStatus = GXSetEnum(hDevice, GX_ENUM_PIXEL_FORMAT, pEnumTemp->nValue);
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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to get parameters of the device
\param   hDevice     
\param   pstCamInfo  

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxTransferDelayDlg::GetCameraParam(GX_DEV_HANDLE hDevice, CAMER_INFO *pstCamInfo)
{
	bool      bIsColorFilter = false;
	int64_t   nWidth         = 0;
	int64_t   nHeight        = 0;
	int64_t   nPayLoadSise   = 0;
	int64_t   nBayer         = 0;
	GX_STATUS emStatus       = GX_STATUS_SUCCESS;

	// Get ImageWidth 
	emStatus = GXGetInt(hDevice, GX_INT_WIDTH, &nWidth);
	VERIFY_STATUS_RET(emStatus);
	pstCamInfo->nImageWidth = nWidth;
	
	// Get ImageHeight
	emStatus = GXGetInt(hDevice, GX_INT_HEIGHT, &nHeight);
	VERIFY_STATUS_RET(emStatus);
	pstCamInfo->nImageHeight = nHeight;

	// Check whether the device supports PixelColorFilter
 	emStatus = GXIsImplemented(hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &bIsColorFilter);
 	VERIFY_STATUS_RET(emStatus);
	pstCamInfo->bIsColorFilter = bIsColorFilter;

	// Get the PixelColorFilter
	if (bIsColorFilter)
	{
		emStatus = GXGetEnum(hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &nBayer);
		VERIFY_STATUS_RET(emStatus);
		pstCamInfo->nBayerLayout = nBayer;
	}
	
	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to allocate memory for showing image.
\param   hDevice     
\param   pstCamInfo  

\return  true : Success, false: Fail
*/
//----------------------------------------------------------------------------------
bool CGxTransferDelayDlg::InitShowImageInfo(GX_DEV_HANDLE hDevice, CAMER_INFO *pstCamInfo)
{
	int i = 0;
	size_t nImageSize = 0;

	// Initialize the BMP header 
	pstCamInfo->pBmpInfo		                    = (BITMAPINFO *)(pstCamInfo->chBmpBuf);
	pstCamInfo->pBmpInfo->bmiHeader.biSize          = sizeof(BITMAPINFOHEADER);
	pstCamInfo->pBmpInfo->bmiHeader.biWidth			= (LONG)pstCamInfo->nImageWidth;
	pstCamInfo->pBmpInfo->bmiHeader.biHeight		= (LONG)pstCamInfo->nImageHeight;
	
	pstCamInfo->pBmpInfo->bmiHeader.biPlanes		= 1;
	pstCamInfo->pBmpInfo->bmiHeader.biBitCount		= pstCamInfo->bIsColorFilter ? 24 : 8; 
	pstCamInfo->pBmpInfo->bmiHeader.biCompression	= BI_RGB;
	pstCamInfo->pBmpInfo->bmiHeader.biSizeImage		= 0;
	pstCamInfo->pBmpInfo->bmiHeader.biXPelsPerMeter	= 0;
	pstCamInfo->pBmpInfo->bmiHeader.biYPelsPerMeter	= 0;
	pstCamInfo->pBmpInfo->bmiHeader.biClrUsed		= 0;
	pstCamInfo->pBmpInfo->bmiHeader.biClrImportant	= 0;
	
	// If using mono cameras,you need to initialize color palette first.
	if (!pstCamInfo->bIsColorFilter)
	{

		for(i = 0; i < 256; i++)
		{
			pstCamInfo->pBmpInfo->bmiColors[i].rgbBlue	   = i;
			pstCamInfo->pBmpInfo->bmiColors[i].rgbGreen	   = i;
			pstCamInfo->pBmpInfo->bmiColors[i].rgbRed	   = i;
			pstCamInfo->pBmpInfo->bmiColors[i].rgbReserved = i;
		}
	}

	if (pstCamInfo->bIsColorFilter)    
	{
		nImageSize = (size_t)(pstCamInfo->nImageHeight * pstCamInfo->nImageWidth * 3);
	}
	else                             
	{
		nImageSize = (size_t)(pstCamInfo->nImageHeight * pstCamInfo->nImageWidth);
	}

	// If the buffer for showing image is not NULL then release it. 
	if (pstCamInfo->pShowBuf != NULL)
	{
		delete[]pstCamInfo->pShowBuf;
		pstCamInfo->pShowBuf = NULL;
	}
	
	// Reallocate memory for showing image
	pstCamInfo->pShowBuf = new BYTE[nImageSize];
	if (pstCamInfo->pShowBuf == NULL)
	{			
		return false;
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to release memory for showing image
\param   ppShowBuf  

\return  void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::UnInitShowImageInfo(BYTE **ppShowBuf)
{
	if (*ppShowBuf != NULL)
	{
		delete[] *ppShowBuf;
		*ppShowBuf = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to start acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::OnStartSnap() 
{
	// TODO: Add your control notification handler code here
	int i = 0;
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	for (i = 0; i < m_nDevNum; i++)
	{
		//Set StreamBufferHandlingMode
		emStatus = GXSetEnum(m_pDeviceHandle[i], GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
		GX_VERIFY(emStatus);

		// Send AcquisitionStart command
		emStatus = GXSendCommand(m_pDeviceHandle[i], GX_COMMAND_ACQUISITION_START);
		GX_VERIFY(emStatus);
	}

	// Set the StartTranEvent of Camera No.0 to signaled, set others to no signal
	SetEvent(m_hCam0StartTranEvent);
	ResetEvent(m_hCam1StartTranEvent);
	ResetEvent(m_hCam2StartTranEvent);
	ResetEvent(m_hCam3StartTranEvent);

	// Set the StopSnapEvent of all the camera to no signal
	ResetEvent(m_hCam0StopSnapEvent);
	ResetEvent(m_hCam1StopSnapEvent);
	ResetEvent(m_hCam2StopSnapEvent);
	ResetEvent(m_hCam3StopSnapEvent);

	m_bIsSnap = true;	
	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief    The ExposureEndEvent callback function of Camera No.0
\param    emEventID		
\param	  pUserParam   
\return   void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFeatureCallbackFun0(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam)
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)pUserParam;
	HANDLE hEevents[2] = {pDlg->m_hCam0StopSnapEvent, pDlg->m_hCam0StartTranEvent};

	// Check whether the feature is ExposureEndEvent
	if (emFeatureID != GX_INT_EVENT_EXPOSUREEND)
	{
        pDlg->PrintInfotoWnd(MsgComposer("The ExposureEndEventID of Camera No.0 is wrong"));
		return;
	}

	switch(::WaitForMultipleObjects(2, hEevents, FALSE, CAM0_WAIT_TIME))
	{

	case WAIT_TIMEOUT:
		pDlg->PrintInfotoWnd(MsgComposer("Camera No.0 WaitTimeout,please stop and restart acquisition"));
		break;

	// The event of transfer delay 
	case WAIT_OBJECT_0 + 1:
		emStatus = GXSendCommand(pDlg->m_pDeviceHandle[CAMERA0], GX_COMMAND_TRANSFER_START);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			pDlg->ShowErrorString(emStatus);
		}
		break;

    // The event of AcquisitionStop
	case WAIT_OBJECT_0:
		break;

	default:	
		break;
	}
}

//---------------------------------------------------------------------------------
/**
\brief    The ExposureEndEvent callback function of Camera No.1
\param    emEventID		
\param	  pUserParam   

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFeatureCallbackFun1(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam)
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)pUserParam;
	HANDLE hEevents[2] = {pDlg->m_hCam1StopSnapEvent, pDlg->m_hCam1StartTranEvent};
	
	// Check whether the feature is ExposureEndEvent
	if (emFeatureID != GX_INT_EVENT_EXPOSUREEND)
	{
		pDlg->PrintInfotoWnd(MsgComposer("The ExposureEndEventID of Camera No.1 is wrong"));
		return;
	}

	switch(::WaitForMultipleObjects(2, hEevents, FALSE, CAM1_WAIT_TIME))
	{

	case WAIT_TIMEOUT:
		pDlg->PrintInfotoWnd(MsgComposer("Camera No.1 WaitTimeout,please stop and restart acquisition"));
		break;

	// The event of transfer delay 
	case WAIT_OBJECT_0 + 1:
		emStatus = GXSendCommand(pDlg->m_pDeviceHandle[CAMERA1], GX_COMMAND_TRANSFER_START);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			pDlg->ShowErrorString(emStatus);
		}
		break;

	// The event of AcquisitionStop
	case WAIT_OBJECT_0:
		break;

	default:	
		break;
	}
}

//---------------------------------------------------------------------------------
/**
\brief    The ExposureEndEvent callback function of Camera No.2
\param    emEventID		
\param	  pUserParam   

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFeatureCallbackFun2(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam)
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)pUserParam;
	HANDLE hEevents[2] = {pDlg->m_hCam2StopSnapEvent, pDlg->m_hCam2StartTranEvent};

	// Check whether the feature is ExposureEndEvent
	if (emFeatureID != GX_INT_EVENT_EXPOSUREEND)
	{
		pDlg->PrintInfotoWnd(MsgComposer("The ExposureEndEventID of Camera No.2 is wrong"));
		return;
	}

	switch(::WaitForMultipleObjects(2, hEevents, FALSE, CAM2_WAIT_TIME))
	{

	case WAIT_TIMEOUT:
		pDlg->PrintInfotoWnd(MsgComposer("Camera No.2 WaitTimeout,please stop and restart acquisition"));
		break;

	// The event of transfer delay 
	case WAIT_OBJECT_0 + 1:
		emStatus = GXSendCommand(pDlg->m_pDeviceHandle[CAMERA2], GX_COMMAND_TRANSFER_START);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			pDlg->ShowErrorString(emStatus);
		}
		break;

	// The event of AcquisitionStop
	case WAIT_OBJECT_0:
		break;

	default:	
		break;
	}
}

//---------------------------------------------------------------------------------
/**
\brief    The ExposureEndEvent callback function of Camera No.3
\param    emEventID		       
\param	  pUserParam    

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFeatureCallbackFun3(GX_FEATURE_ID_CMD emFeatureID, void* pUserParam)
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)pUserParam;
	HANDLE hEevents[2] = {pDlg->m_hCam3StopSnapEvent, pDlg->m_hCam3StartTranEvent};

	// Check whether the feature is ExposureEndEvent
	if (emFeatureID != GX_INT_EVENT_EXPOSUREEND)
	{
		pDlg->PrintInfotoWnd(MsgComposer("The ExposureEndEventID of Camera No.3 is wrong"));
		return;
	}

	switch(::WaitForMultipleObjects(2, hEevents, FALSE, CAM3_WAIT_TIME))
	{

	case WAIT_TIMEOUT:
		pDlg->PrintInfotoWnd(MsgComposer("Camera No.3 WaitTimeout,please stop and restart acquisition"));
		break;

	// The event of transfer delay 
	case WAIT_OBJECT_0 + 1:
		emStatus = GXSendCommand(pDlg->m_pDeviceHandle[CAMERA3], GX_COMMAND_TRANSFER_START);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			pDlg->ShowErrorString(emStatus);
		}
		break;

	// The event of AcquisitionStop
	case WAIT_OBJECT_0:
		break;

	default:	
		break;
	}	
}

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to convert the image format and draw image
\param   pFrame  

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFrameCallbackFun0(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)(pFrame->pUserParam);

	// Check the current image frame status 
	if (pFrame->status != 0)
	{
        pDlg->PrintInfotoWnd(MsgComposer("Camera No.0 has incomplete frames"));
	}

	// Enable camera No.1
	SetEvent(pDlg->m_hCam1StartTranEvent);

	int		    nID       = 0;             
	BYTE	    *pShowBuf = NULL;          
	BITMAPINFO	*pBmpInfo = NULL;          
	int64_t	    nBayer    = 0;          
	int		    nWidth    = 0;             
	int         nHeight   = 0;           
	int         i         = 0;            

	pShowBuf = pDlg->m_pstCamInfo[nID].pShowBuf;
 	nWidth   = (int)(pDlg->m_pstCamInfo[nID].nImageWidth);
 	nHeight  = (int)(pDlg->m_pstCamInfo[nID].nImageHeight);
 	nBayer   = pDlg->m_pstCamInfo[nID].nBayerLayout;
 	pBmpInfo = pDlg->m_pstCamInfo[nID].pBmpInfo;

	if (pDlg->m_pstCamInfo[nID].bIsColorFilter)
	{
		// If the acquired image is color format,convert it to RGB 
		DxRaw8toRGB24((BYTE *)pFrame->pImgBuf, pShowBuf, nWidth, nHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayer), TRUE);
	}
	else
	{
		// If the acquired image is mono format,you must flip the image data for showing.
		for(i = 0; i < nHeight; i++)
		{
			memcpy(pShowBuf + i * nWidth, (BYTE *)pFrame->pImgBuf + (nHeight - i - 1) * nWidth, nWidth);
		}		
	}
	
	pDlg->DrawImage(pShowBuf, nWidth, nHeight, pBmpInfo, pDlg->m_hDC[0], pDlg->m_objRect[0]);
}

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to convert the image format and draw image
\param   pFrame  

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFrameCallbackFun1(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)(pFrame->pUserParam);

	// Check the current image frame status 
	if (pFrame->status != 0)
	{
		pDlg->PrintInfotoWnd(MsgComposer("Camera No.1 has incomplete frames"));
	}

	// Enable camera No.2
	SetEvent(pDlg->m_hCam2StartTranEvent);
	
	int		    nID       = 1;             // ID number
	BYTE	    *pShowBuf = NULL;          // Converted image buffer
	BITMAPINFO	*pBmpInfo = NULL;          // Bitmap information
	int64_t	    nBayer    = 0;             // PixelColorFilter
	int		    nWidth    = 0;             // ImgWidth
	int         nHeight   = 0;             // ImgHeight 
	int         i         = 0;             // Loop variable
	
	pShowBuf = pDlg->m_pstCamInfo[nID].pShowBuf;
	nWidth   = (int)(pDlg->m_pstCamInfo[nID].nImageWidth);
	nHeight  = (int)(pDlg->m_pstCamInfo[nID].nImageHeight);
	nBayer   = pDlg->m_pstCamInfo[nID].nBayerLayout;
	pBmpInfo = pDlg->m_pstCamInfo[nID].pBmpInfo;
	
	if (pDlg->m_pstCamInfo[nID].bIsColorFilter)
	{
		// Convert the raw8 image to RGB for showing image if color camera has been used. 
		DxRaw8toRGB24((BYTE *)pFrame->pImgBuf, pShowBuf, nWidth, nHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayer), TRUE);
	}
	else
	{
		// Show image after flipping image data if mono camera has been used.
		for(i = 0; i < nHeight; i++)
		{
			memcpy(pShowBuf + i * nWidth, (BYTE *)pFrame->pImgBuf + (nHeight - i - 1) * nWidth, nWidth);
		}		
	}
	
	pDlg->DrawImage(pShowBuf, nWidth, nHeight, pBmpInfo, pDlg->m_hDC[1], pDlg->m_objRect[1]);
}

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to convert the image format and draw image
\param   pFrame 

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFrameCallbackFun2(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)(pFrame->pUserParam);

	// Check the current image frame status 
	if (pFrame->status != 0)
	{
		pDlg->PrintInfotoWnd(MsgComposer("Camera No.2 has incomplete frames"));
	}

	// Enable camera No.3
	SetEvent(pDlg->m_hCam3StartTranEvent);
		
	int		    nID       = 2;             // ID number
	BYTE	    *pShowBuf = NULL;          // Converted image buffer
	BITMAPINFO	*pBmpInfo = NULL;          // Bitmap information
	int64_t	    nBayer    = 0;             // PixelColorFilter
	int		    nWidth    = 0;             // ImgWidth
	int         nHeight   = 0;             // ImgHeight 
	int         i         = 0;             // Loop variable
	
	pShowBuf = pDlg->m_pstCamInfo[nID].pShowBuf;
	nWidth   = (int)(pDlg->m_pstCamInfo[nID].nImageWidth);
	nHeight  = (int)(pDlg->m_pstCamInfo[nID].nImageHeight);
	nBayer   = pDlg->m_pstCamInfo[nID].nBayerLayout;
	pBmpInfo = pDlg->m_pstCamInfo[nID].pBmpInfo;
		
	if (pDlg->m_pstCamInfo[nID].bIsColorFilter)
	{
		// Convert the raw8 image to RGB for showing image if color camera has been used. 
		DxRaw8toRGB24((BYTE *)pFrame->pImgBuf, pShowBuf, nWidth, nHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayer), TRUE);
	}
	else
	{
		// Show image after flipping image data if mono camera has been used.
		for(i = 0; i < nHeight; i++)
		{
			memcpy(pShowBuf + i * nWidth, (BYTE *)pFrame->pImgBuf + (nHeight - i - 1) * nWidth, nWidth);
		}		
	}
	
	pDlg->DrawImage(pShowBuf, nWidth, nHeight, pBmpInfo, pDlg->m_hDC[2], pDlg->m_objRect[2]);
}

//---------------------------------------------------------------------------------
/**
\brief   This is a callback function that demonstrates how to convert the image format and draw image   
\param   pFrame  

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxTransferDelayDlg::OnFrameCallbackFun3(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxTransferDelayDlg *pDlg = (CGxTransferDelayDlg*)(pFrame->pUserParam);

	// Check the current image frame status 
	if (pFrame->status != 0)
	{
		pDlg->PrintInfotoWnd(MsgComposer("Camera No.3 has incomplete frames"));
	}

	// Enable camera No.0
	SetEvent(pDlg->m_hCam0StartTranEvent);
	
	int		    nID       = 3;             // ID number
	BYTE	    *pShowBuf = NULL;          // Converted image buffer
	BITMAPINFO	*pBmpInfo = NULL;          // Bitmap information
	int64_t	    nBayer    = 0;             // PixelColorFilter
	int		    nWidth    = 0;             // ImgWidth
	int         nHeight   = 0;             // ImgHeight 
	int         i         = 0;             // Loop variable
	
	pShowBuf = pDlg->m_pstCamInfo[nID].pShowBuf;
	nWidth   = (int)(pDlg->m_pstCamInfo[nID].nImageWidth);
	nHeight  = (int)(pDlg->m_pstCamInfo[nID].nImageHeight);
	nBayer   = pDlg->m_pstCamInfo[nID].nBayerLayout;
	pBmpInfo = pDlg->m_pstCamInfo[nID].pBmpInfo;
	
	if (pDlg->m_pstCamInfo[nID].bIsColorFilter)
	{
		// Convert the raw8 image to RGB for showing image if color camera has been used. 
		DxRaw8toRGB24((BYTE *)pFrame->pImgBuf, pShowBuf, nWidth, nHeight, RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(nBayer), TRUE);
	}
	else
	{
		// Show image after flipping image data if mono camera has been used.
		for(i = 0; i < nHeight; i++)
		{
			memcpy(pShowBuf + i * nWidth, (BYTE *)pFrame->pImgBuf + (nHeight - i - 1) * nWidth, nWidth);
		}		
	}
	
	pDlg->DrawImage(pShowBuf, nWidth, nHeight, pBmpInfo, pDlg->m_hDC[3], pDlg->m_objRect[3]);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to draw image in the window
\param   pImageBuf  
\param   nWidth     
\param   nHeight     
\param   pBmpInfo   
\param   pWnd       

\return  void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo, HDC hDC, RECT objRect)
{
	if ((pImageBuf == NULL) || (pBmpInfo == NULL) || (hDC == NULL))
	{
		return;
	}

	int nWndWidth  = objRect.right - objRect.left;    
	int nWndHeight = objRect.bottom - objRect.top;     

	// You could use the following code to show image. 
	::SetStretchBltMode(hDC, COLORONCOLOR);
	::StretchDIBits(hDC,
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
\brief  This function demonstrates how to stop acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::OnStopSnap() 
{
	// TODO: Add your control notification handler code here
	int i = 0;
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Set the StopSnapEvent of all the cameras to signaled
	SetEvent(m_hCam0StopSnapEvent);
	SetEvent(m_hCam1StopSnapEvent);
	SetEvent(m_hCam2StopSnapEvent);
	SetEvent(m_hCam3StopSnapEvent);

	for (i = 0; i < m_nDevNum; i++)
	{
		// Send AcquisitionStop command
		emStatus = GXSendCommand(m_pDeviceHandle[i], GX_COMMAND_ACQUISITION_STOP);
		GX_VERIFY(emStatus);

		// Flush image queues to clear out-of-date images
		emStatus = GXFlushEvent(m_pDeviceHandle[i]);
		GX_VERIFY(emStatus);
	}

	m_bIsSnap = false;
	UpdateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.

\return void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::OnCloseDevice() 
{
	// TODO: Add your control notification handler code here
	int i = 0;
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Set the StopSnapEvent of all the cameras to signaled
		SetEvent(m_hCam0StopSnapEvent);
		SetEvent(m_hCam1StopSnapEvent);
		SetEvent(m_hCam2StopSnapEvent);
		SetEvent(m_hCam3StopSnapEvent);

		for (i = 0; i < m_nDevNum; i++)
		{
			// Send AcquisitionStop command
		    emStatus = GXSendCommand(m_pDeviceHandle[i], GX_COMMAND_ACQUISITION_STOP);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				// TODO: Add your error handler code here  
			}

			// Flush image queues to clear out-of-date images

		    emStatus = GXFlushEvent(m_pDeviceHandle[i]);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				// TODO: Add your error handler code here  
			}
		}
	}
	m_bIsSnap = false;


	for (i = 0; i < m_nDevNum; i++)
	{
		// Unregister the CaptureCallback function
		emStatus = GXUnregisterCaptureCallback(m_pDeviceHandle[i]);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}
		
		// Unregister the ExposureEvent callback function
		emStatus = GXUnregisterFeatureCallback(m_pDeviceHandle[i], GX_INT_EVENT_EXPOSUREEND, m_pCBHandle[i]);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}

		emStatus = GXCloseDevice(m_pDeviceHandle[i]);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			// TODO: Add your error handler code here  
		}
		m_pDeviceHandle[i] = NULL;

		// Release memory for showing image
		UnInitShowImageInfo(&m_pstCamInfo[i].pShowBuf);
	}
	m_bIsOpen = false;

	UpdateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize camera resources

\return ture: Success; false:Fail
*/
//----------------------------------------------------------------------------------
bool CGxTransferDelayDlg::InitResources()
{
	m_pDeviceHandle = new GX_DEV_HANDLE[MAX_NUM_CAMERA];
	if (m_pDeviceHandle == NULL)
	{
		return false;
	}

	// The pointer of the exposure callback function
	m_pCBHandle = new GX_FEATURE_CALLBACK_HANDLE[MAX_NUM_CAMERA];
	if (m_pCBHandle == NULL)
	{
		ReleaseResource();
		return false;
	}
	
	// The structure of camera information 
	m_pstCamInfo = new CAMER_INFO[MAX_NUM_CAMERA];
	if (m_pstCamInfo == NULL)
	{
		ReleaseResource();
		return false;
	}

	// Create Camera No.0 StartTransferEvent
	m_hCam0StartTranEvent = CreateEvent(NULL, FALSE, FALSE, "Camer0StartTranEvent");
	if (m_hCam0StartTranEvent == NULL)
	{
		ReleaseResource();
		return false;
	}
   	
	// Create Camera No.1 StartTransferEvent
	m_hCam1StartTranEvent = CreateEvent(NULL, FALSE, FALSE, "Camer1StartTranEvent");
	if (m_hCam1StartTranEvent == NULL)
	{
		ReleaseResource();
		return false;
	}
	
	// Create Camera No.2 StartTransferEvent
	m_hCam2StartTranEvent = CreateEvent(NULL, FALSE, FALSE, "Camer2StartTranEvent");
	if (m_hCam2StartTranEvent == NULL)
	{
		ReleaseResource();
		return false;
	}
	
	// Create Camera No.3 StartTransferEvent
	m_hCam3StartTranEvent = CreateEvent(NULL, FALSE, FALSE, "Camer3StartTranEvent");
	if (m_hCam3StartTranEvent == NULL)
	{
		ReleaseResource();
		return false;
	}	

	// Create Camera No.0 StopSnapEvent
	m_hCam0StopSnapEvent = CreateEvent(NULL, TRUE, FALSE, "Camer0StopSnapEvent");
	if (m_hCam0StopSnapEvent == NULL)
	{
		ReleaseResource();
		return false;
	}

	// Create Camera No.1 StopSnapEvent
	m_hCam1StopSnapEvent = CreateEvent(NULL, TRUE, FALSE, "Camer1StopSnapEvent");
	if (m_hCam1StopSnapEvent == NULL)
	{
		ReleaseResource();
		return false;
	}

	// Create Camera No.2 StopSnapEvent
	m_hCam2StopSnapEvent = CreateEvent(NULL, TRUE, FALSE, "Camer2StopSnapEvent");
	if (m_hCam2StopSnapEvent == NULL)
	{
		ReleaseResource();
		return false;
	}

	// Create Camera No.3 StopSnapEvent
	m_hCam3StopSnapEvent = CreateEvent(NULL, TRUE, FALSE, "Camer3StopSnapEvent");
	if (m_hCam3StopSnapEvent == NULL)
	{
		ReleaseResource();
		return false;
	}		
	
	for (int i = 0; i < MAX_NUM_CAMERA; i++)
	{
		m_pDeviceHandle[i]             = NULL;
		m_pCBHandle[i]                 = NULL;
		m_pstCamInfo[i].bIsColorFilter = false;
		m_pstCamInfo[i].pBmpInfo       = NULL;
		m_pstCamInfo[i].pShowBuf       = NULL;
        m_pstCamInfo[i].nBayerLayout   = 0;
		m_pstCamInfo[i].nImageHeight   = 0;
		m_pstCamInfo[i].nImageWidth    = 0;
		memset(m_pstCamInfo[i].chBmpBuf, 0, sizeof(m_pstCamInfo[i].chBmpBuf));
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release resources

\return void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::ReleaseResource()
{
	// Release the pointer for device handle 
	if(m_pDeviceHandle != NULL)
	{
		delete[]m_pDeviceHandle;
		m_pDeviceHandle = NULL;
	}

	// Release the pointer for the exposure callback function
	if (m_pCBHandle != NULL)
	{
		delete[]m_pCBHandle;
		m_pCBHandle = NULL;
	}

	// Release the pointer for the camera data structure 
	if(m_pstCamInfo != NULL)
	{
		for(int i = 0; i < m_nDevNum; i++)
		{
			if(m_pstCamInfo[i].pShowBuf != NULL)
			{
				delete[]m_pstCamInfo[i].pShowBuf;
				m_pstCamInfo[i].pShowBuf = NULL;
			}
		}
		delete[]m_pstCamInfo;
        m_pstCamInfo = NULL;
	}

	// Release the handle for TransferDelayEvent 
    if (m_hCam0StartTranEvent != NULL)
    {
        CloseHandle(m_hCam0StartTranEvent);
        m_hCam0StartTranEvent = NULL;
    }
	if (m_hCam1StartTranEvent != NULL)
    {
        CloseHandle(m_hCam1StartTranEvent);
        m_hCam1StartTranEvent = NULL;
    }
	if (m_hCam2StartTranEvent != NULL)
    {
        CloseHandle(m_hCam2StartTranEvent);
        m_hCam2StartTranEvent = NULL;
    }
	if (m_hCam3StartTranEvent != NULL)
    {
        CloseHandle(m_hCam3StartTranEvent);
        m_hCam3StartTranEvent = NULL;
    }

	// Release the handle for StopSnapEvent
	if (m_hCam0StopSnapEvent != NULL)
	{
		CloseHandle(m_hCam0StopSnapEvent);
		m_hCam0StopSnapEvent = NULL;
	}
	if (m_hCam1StopSnapEvent != NULL)
	{
		CloseHandle(m_hCam1StopSnapEvent);
		m_hCam1StopSnapEvent = NULL;
	}
	if (m_hCam2StopSnapEvent != NULL)
	{
		CloseHandle(m_hCam2StopSnapEvent);
		m_hCam2StopSnapEvent = NULL;
	}
	if (m_hCam3StopSnapEvent != NULL)
	{
		CloseHandle(m_hCam3StopSnapEvent);
		m_hCam3StopSnapEvent = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  

\return void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::ShowErrorString(GX_STATUS emErrorStatus)
{
	char      *pchErrorInfo = NULL;
	size_t    nSize         = 0;
	GX_STATUS emStatus      = GX_STATUS_SUCCESS;
	
	// Get the length of error information and allocate memory.
	emStatus = GXGetLastError(&emErrorStatus, NULL, &nSize);
	pchErrorInfo = new char[nSize];
	if (pchErrorInfo == NULL)
	{
		return;
	}
	
	emStatus = GXGetLastError(&emErrorStatus, pchErrorInfo, &nSize);
	
	// Get error information and show them to user 
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

//----------------------------------------------------------------------------------
/**
\brief   Update UI

\return  
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::UpdateUI()
{
	GetDlgItem(IDC_OPEN_DEVICE)->EnableWindow(!m_bIsOpen);
	GetDlgItem(IDC_START_SNAP)->EnableWindow(m_bIsOpen && (!m_bIsSnap));
	GetDlgItem(IDC_STOP_SNAP)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to print log information in the window
\param  strInfo 

\return void
*/
//----------------------------------------------------------------------------------
void CGxTransferDelayDlg::PrintInfotoWnd(string &strInfo)
{
	SYSTEMTIME system;
	GetLocalTime(&system);
	string strTime = MsgComposer("%02dh:%02dm:%02ds:%03dms ",system.wHour, system.wMinute, system.wSecond, system.wMilliseconds);
	strInfo = strTime.append(strInfo);
	
	if (m_pLogWnd != NULL)
	{
		CEdit *pEdit = (CEdit*)m_pLogWnd;
		int textLen = pEdit->GetWindowTextLength();
		
		if (pEdit->GetLineCount() > 300)
		{
			pEdit->Clear();
			pEdit->SetWindowText(_T(""));
			textLen = pEdit->GetWindowTextLength();
		}
		
		pEdit->SetSel(textLen, -1);	
		strInfo.append("\r\n");
		pEdit->ReplaceSel((LPCTSTR)strInfo.c_str());
		
		if(textLen > 0)
		{
			pEdit->SetSel((textLen - pEdit->GetLineCount() + 1), -1);
		}
		else
		{
			pEdit->SetSel(textLen, -1);
		}
	}
}


BOOL CGxTransferDelayDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
		return TRUE;
	}   
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_ESCAPE))   
	{   
		return  TRUE; 
	}
	
	return CDialog::PreTranslateMessage(pMsg);
}