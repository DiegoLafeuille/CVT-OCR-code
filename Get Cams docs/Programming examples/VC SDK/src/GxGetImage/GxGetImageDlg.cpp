// GxGetImageDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxGetImage.h"
#include "GxGetImageDlg.h"

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
	CString	m_strLegalCopyRight;
	CString	m_strProductVersion;
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
	m_strLegalCopyRight = _T("");
	m_strProductVersion = _T("");
	//}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CAboutDlg)
	DDX_Text(pDX, IDC_STATIC_LEGALCOPYRIGHT, m_strLegalCopyRight);
	DDX_Text(pDX, IDC_STATIC_PRODUCTVERSION, m_strProductVersion);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
	//{{AFX_MSG_MAP(CAboutDlg)
		// No message handlers
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxGetImageDlg dialog

CGxGetImageDlg::CGxGetImageDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxGetImageDlg::IDD, pParent)
	, m_bIsColorFilter(false)
	, m_bIsOpen(false)
	, m_bIsSnap(false)
	, m_nPayLoadSize(0)
	, m_nPixelColorFilter(GX_COLOR_FILTER_NONE)
	, m_nImageWidth(0)
	, m_nImageHeight(0)
	, m_pBmpInfo(NULL)
	, m_hDevice(NULL)
    , m_pImgBuffer(NULL)
	, m_pWnd(NULL)
	, m_hDC(NULL)
{
	//{{AFX_DATA_INIT(CGxGetImageDlg)
	m_nEditTimeOut = 500;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32

  
	memset(&m_stFrameData,0,sizeof(GX_FRAME_DATA)); 
	memset(m_chBmpBuf, 0 ,sizeof(m_chBmpBuf));

	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CGxGetImageDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxGetImageDlg)
	DDX_Text(pDX, IDC_EDIT_TIME_OUT, m_nEditTimeOut);
	DDV_MinMaxUInt(pDX, m_nEditTimeOut, 0, 3600000);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxGetImageDlg, CDialog)
	//{{AFX_MSG_MAP(CGxGetImageDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_BTN_START_SNAP,OnBnClickedBtnStartSnap)
	ON_BN_CLICKED(IDC_BTN_STOP_SNAP,OnBnClickedBtnStopSnap)
	ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE,OnBnClickedBtnOpenDevice)
	ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE,OnBnClickedBtnCloseDevice)
	ON_BN_CLICKED(IDC_BTN_SEND_SOFT_COMMAND,OnBnClickedBtnSendSoftCommand)
	ON_BN_CLICKED(IDC_BTN_CLEAN_TIME,OnBnClickedBtnCleanTime)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxGetImageDlg message handlers
BOOL CGxGetImageDlg::OnInitDialog()
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
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Before using any GxIAPI methods, the GxIAPI must be initialized.   
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0); 
	}

	//Get the pointer and DC handle for showing image
	m_pWnd = GetDlgItem(IDC_STATIC_SHOW_WND);
	m_hDC  = ::GetDC(m_pWnd->m_hWnd);


	UpdateUI();
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxGetImageDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char strFileName[MAX_PATH] = {'\0'};
        GetModuleFileName(NULL, strFileName, MAX_PATH);
        CFileVersion fv(strFileName);
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
void CGxGetImageDlg::OnPaint() 
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
HCURSOR CGxGetImageDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to open device.            

\return Success:GX_STATUS_SUCCESS  Else:Error
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxGetImageDlg::OpenDevice()
{
	GX_STATUS      emStatus    = GX_STATUS_SUCCESS;	
	GX_OPEN_PARAM  stOpenParam;

	// Open the first found device 
	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_INDEX;
	stOpenParam.pszContent = "1";

	
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.
 
\return Success:GX_STATUS_SUCCESS  Else:Error
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxGetImageDlg::CloseDevice()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;

	// Release resources for showing image 
    UnPrepareForShowImg();
	
	emStatus = GXCloseDevice(m_hDevice);
		
	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxGetImageDlg::PrepareForShowImg()
{	
	bool bRet = false;

	//Allocate memory for getting image
	m_stFrameData.pImgBuf     = new BYTE[(size_t)m_nPayLoadSize];
	if (m_stFrameData.pImgBuf == NULL)
	{
		bRet = false;
	}

	if (m_bIsColorFilter)   
	{
		// Allocate buffer for showing color image.
		bRet = PrepareForShowColorImg();
	}
	else  
	{
		// Allocate buffer for showing mono image.
		bRet = PrepareForShowMonoImg();
	}

	// If fail to allocate one of the buffers then release buffers have been allocated. 
	if (!bRet)
	{
		UnPrepareForShowImg();
	}

	return bRet;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing color image.
                                                               
\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxGetImageDlg::PrepareForShowColorImg()
{	
	//--------------------------------------------------------------------
	//----------------------Initialize bitmap header----------------------
	m_pBmpInfo								= (BITMAPINFO *)m_chBmpBuf;
	m_pBmpInfo->bmiHeader.biSize			= sizeof(BITMAPINFOHEADER);
	m_pBmpInfo->bmiHeader.biWidth			= (LONG)m_nImageWidth;
	m_pBmpInfo->bmiHeader.biHeight			= (LONG)m_nImageHeight;	

	m_pBmpInfo->bmiHeader.biPlanes			= 1;
	m_pBmpInfo->bmiHeader.biBitCount		= 24; 
	m_pBmpInfo->bmiHeader.biCompression		= BI_RGB;                            
	m_pBmpInfo->bmiHeader.biSizeImage		= 0;
	m_pBmpInfo->bmiHeader.biXPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biYPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biClrUsed			= 0;
	m_pBmpInfo->bmiHeader.biClrImportant	= 0;


	//--------------------------------------------------------------------
	//-------------------Allocate buffer for showing image ---------------

	// Allocate buffer for showing converted image.

	m_pImgBuffer = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight * 3)];
	if (m_pImgBuffer == NULL)
	{
		return false;
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing mono image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxGetImageDlg::PrepareForShowMonoImg()
{	
	//--------------------------------------------------------------------
	//----------------------Initialize bitmap header----------------------
	m_pBmpInfo								= (BITMAPINFO *)m_chBmpBuf;
	m_pBmpInfo->bmiHeader.biSize			= sizeof(BITMAPINFOHEADER);
	m_pBmpInfo->bmiHeader.biWidth			= (LONG)m_nImageWidth;
	m_pBmpInfo->bmiHeader.biHeight			= (LONG)m_nImageHeight;	

	m_pBmpInfo->bmiHeader.biPlanes			= 1;
	m_pBmpInfo->bmiHeader.biBitCount		= 8;  
	m_pBmpInfo->bmiHeader.biCompression		= BI_RGB;
	m_pBmpInfo->bmiHeader.biSizeImage		= 0;
	m_pBmpInfo->bmiHeader.biXPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biYPelsPerMeter	= 0;
	m_pBmpInfo->bmiHeader.biClrUsed			= 0;
	m_pBmpInfo->bmiHeader.biClrImportant	= 0;

	// If using mono cameras,you need to initialize color palette first.
	for(int i=0;i<256;i++)
	{
		m_pBmpInfo->bmiColors[i].rgbBlue	 = i;
		m_pBmpInfo->bmiColors[i].rgbGreen	 = i;
		m_pBmpInfo->bmiColors[i].rgbRed		 = i;
		m_pBmpInfo->bmiColors[i].rgbReserved = i;
	}

	//---------------------------------------------------------------------
	//-------------------Allocate buffer for showing image data------------
	// Allocate memory for showing mono images.
	m_pImgBuffer = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
	if (m_pImgBuffer == NULL)
	{
		return false;
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release memory for showing image.

\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::UnPrepareForShowImg()
{
	if (m_stFrameData.pImgBuf != NULL)
	{
		delete []m_stFrameData.pImgBuf;
		m_stFrameData.pImgBuf = NULL;
	}

	if (m_pImgBuffer != NULL)
	{
		delete []m_pImgBuffer;
		m_pImgBuffer = NULL;
	}
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device 
                                          

\return  void
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxGetImageDlg::InitDevice()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);
	
    //Set the TriggerMode on
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_ON);
	VERIFY_STATUS_RET(emStatus);


	// If the device support the pixel format of"GX_PIXEL_FORMAT_BAYER_GR8",
	// the following code can be used to set the pixel format to GX_PIXEL_FORMAT_BAYER_GR8:
	// emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_GR8);
	// VERIFY_STATUS_RET(emStatus);

	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
    emStatus = SetPixelFormat8bit(); 
	VERIFY_STATUS_RET(emStatus);
 
	//Set the TriggerSource to SoftTrigger
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_SOURCE, GX_TRIGGER_SOURCE_SOFTWARE);

	return emStatus;
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\return  emStatus GX_STATUS_SUCCESS:Success,else:Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxGetImageDlg::SetPixelFormat8bit()
{
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	int64_t   nPixelSize  = 0;
	uint32_t  nEnmuEntry  = 0;
	size_t    nBufferSize = 0;
	BOOL      bIs8bit     = TRUE;

    GX_ENUM_DESCRIPTION  *pEnumDescription = NULL;
	GX_ENUM_DESCRIPTION  *pEnumTemp        = NULL;

	// Get the feature PixelSize, this feature indicates the depth of the pixel values in the acquired images in bits per pixel
	emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_SIZE, &nPixelSize);
    VERIFY_STATUS_RET(emStatus);

	// If the PixelSize is 8bit then return, or set the PixelSize to 8bit.
	if (nPixelSize == GX_PIXEL_SIZE_BPP8)
	{
		return GX_STATUS_SUCCESS;
	}
	else
	{                                
		// Get the enumeration entry of the pixel format the device supports.
		emStatus = GXGetEnumEntryNums(m_hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
		VERIFY_STATUS_RET(emStatus);

		// Allocate memory for getting the enumeration entry of the pixel format.
		nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
		pEnumDescription = new GX_ENUM_DESCRIPTION[nEnmuEntry];

		emStatus         = GXGetEnumDescription(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
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
		
		for (uint32_t i = 0; i<nEnmuEntry; i++)
		{
			if ((pEnumDescription[i].nValue & GX_PIXEL_8BIT) == GX_PIXEL_8BIT)
			{
				emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription[i].nValue);
				break;
			}
		}	

		// Release memory
		if(pEnumDescription != NULL)
		{
			delete []pEnumDescription;
			pEnumDescription = NULL;
		}
	}

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to show image.
\param  pFrameData  [in] A pointer for image data  

\return void
*/
//----------------------------------------------------------------------------------
void  CGxGetImageDlg::ShowImage(GX_FRAME_DATA* pFrameData)
{
	// Check whether the image is completed
	if (pFrameData->nStatus != 0)    
	{
		return;
	}

	//If the acquired image is color format,convert it to RGB 
	if (m_bIsColorFilter)
	{
		//Convert the raw8 image to RGB for showing 
		DxRaw8toRGB24((BYTE*)pFrameData->pImgBuf, m_pImgBuffer, (VxUint32)m_nImageWidth, (VxUint32)m_nImageHeight, RAW2RGB_NEIGHBOUR,
			DX_PIXEL_COLOR_FILTER(m_nPixelColorFilter), TRUE);
	}
	else   
	{
		// If the acquired image is mono format,you must flip the image data for showing.
		for(int i =0;i <m_nImageHeight;i++)
		{
			memcpy(m_pImgBuffer+i*m_nImageWidth, (BYTE *)(pFrameData->pImgBuf)+(m_nImageHeight-i-1)*m_nImageWidth,(size_t)m_nImageWidth);
		}
	}

	//Draw a image in the GUI 
	DrawImg(m_pImgBuffer, m_nImageWidth, m_nImageHeight);
}

//----------------------------------------------------------------------------------
/**
\brief  Draw image in the window
\param  pImageBuf    [in] Pointer for image buffer 
\param  nImageWidth  [in] Image width
\param  nImageHeight [in] Image height

\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::DrawImg(void *pImageBuf, int64_t nImageWidth, int64_t nImageHeight)
{
	int nWndWidth  = 0;
	int nWndHeight = 0;

	// Initialize the parameters for drawing image.
	RECT objRect;
	m_pWnd->GetClientRect(&objRect);
	nWndWidth  = objRect.right - objRect.left;
	nWndHeight = objRect.bottom - objRect.top;

	// You could use the following code to show image. 
	::SetStretchBltMode(m_hDC, COLORONCOLOR);
	::StretchDIBits(m_hDC,
		            0,						
		            0,
		            nWndWidth,
		            nWndHeight,
		            0,
		            0,
		           (int)nImageWidth,
		           (int)nImageHeight,
		            pImageBuf,
		            m_pBmpInfo,
		            DIB_RGB_COLORS,
		            SRCCOPY
		            );	
}

//---------------------------------------------------------------------------------
/**
\brief   Update UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::UpdateUI()
{
	GetDlgItem(IDC_EDIT_TIME_OUT)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_BTN_SEND_SOFT_COMMAND)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bIsOpen);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(m_bIsOpen && !m_bIsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(m_bIsSnap);
    GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);
}

BOOL CGxGetImageDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	CWnd  *pWnd     = NULL;
	int   nCtrlID   = 0;
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
		pWnd = GetFocus();
		nCtrlID = pWnd->GetDlgCtrlID();
		switch(nCtrlID)
		{
		case IDC_EDIT_TIME_OUT:
				::SetFocus(NULL);
				break;
		default:
				break;
		}
			
		return TRUE;
	}   
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_ESCAPE))   
	{   
		return  TRUE; 
	}
			
	return CDialog::PreTranslateMessage(pMsg);
}

void CGxGetImageDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_ERROR;

	// If the device has been opened then close it.
	if (m_bIsOpen)
	{
		emStatus = CloseDevice();
	}
    
	
	::ReleaseDC(m_pWnd->m_hWnd, m_hDC);

	// Release GxIAPI
	emStatus = GXCloseLib();

	CDialog::OnClose();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  status  Error 

\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::ShowErrorString(GX_STATUS emErrorStatus)
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
	if (NULL != pchErrorInfo)
	{
		delete[] pchErrorInfo;
		pchErrorInfo = NULL;
	}
}

//----------------------------------------------------------------------------------
/*
\brief  This function demonstrates how to start acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::OnBnClickedBtnStartSnap()
{
	// TODO: Add your control notification handler code here

	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevice, GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	GX_VERIFY(emStatus);

	// Send AcquisitionStart command
	emStatus = GXSendCommand(m_hDevice,GX_COMMAND_ACQUISITION_START);
	GX_VERIFY(emStatus);

	m_bIsSnap = true;

	UpdateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop acquisition
\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::OnBnClickedBtnStopSnap()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Send AcquisitionStop command 
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);

	m_bIsSnap = false;

	//Reset time statistic to original state
	m_objStatistic.Reset();

	UpdateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.

\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::OnBnClickedBtnOpenDevice()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	bool      bRet     = true;
	uint32_t  nDevNum  = 0;

    // Enumerate all camera devices
	emStatus = GXUpdateDeviceList(&nDevNum, 1000);
	GX_VERIFY(emStatus);

	// Check if found any device

	if (nDevNum <= 0)
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

	//Open the first found device 
	emStatus = OpenDevice();
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

	emStatus = InitDevice();
	GX_VERIFY(emStatus);

	emStatus = GetDeviceInitParam();
    GX_VERIFY(emStatus);

	//Allocate resources for showing image 

	bRet = PrepareForShowImg();
	if (!bRet)
	{
        MessageBox("Fail to allocate resource for showing image!");
		return;
	}


	m_bIsOpen = true;

	UpdateUI();

	UpdateData(FALSE);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to get the parameters of the device

\return void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxGetImageDlg::GetDeviceInitParam()
{
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	bool      bIsImplemented = false;

	// Check whether the current camera supports PixelColorFilter feature
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &bIsImplemented);
	VERIFY_STATUS_RET(emStatus);
	m_bIsColorFilter = bIsImplemented;

	if(bIsImplemented)
	{
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter);
		VERIFY_STATUS_RET(emStatus);
	}

	// Get image width
	emStatus = GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth);
	VERIFY_STATUS_RET(emStatus);

	// Get image height
	emStatus = GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);
	VERIFY_STATUS_RET(emStatus);

	// Get image size
    emStatus = GXGetInt(m_hDevice,GX_INT_PAYLOAD_SIZE,&m_nPayLoadSize);

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.
\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::OnBnClickedBtnCloseDevice()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if(m_bIsSnap)
	{
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
        if (emStatus != GX_STATUS_SUCCESS)
        {
			// TODO: Add your error handler code here  
        }
		m_bIsSnap = false;
	}

	emStatus = CloseDevice();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here  
	}

	m_bIsOpen = false;
	m_hDevice = NULL;

	UpdateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to send TriggerSoftware command,acquire image and show image.

\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::OnBnClickedBtnSendSoftCommand()
{
	// TODO: Add your control notification handler code here
	if (!UpdateData(TRUE))
	{
		m_nEditTimeOut = 500;
		return;
	}

	GX_STATUS emStatus  = GX_STATUS_ERROR;
	double dElapsedtime = 0;

	//Flush image queues to clear out-of-date images
	emStatus = GXFlushQueue(m_hDevice);
	GX_VERIFY(emStatus);

    //Send TriggerSoftware commands
	emStatus = GXSendCommand(m_hDevice,GX_COMMAND_TRIGGER_SOFTWARE);
	GX_VERIFY(emStatus);

	//Start stopwatch
	m_objStopTime.Start();

	//Get image
	emStatus = GXGetImage(m_hDevice, &m_stFrameData, m_nEditTimeOut);

	//Stop stopwatch and get the ElapsedTime
	dElapsedtime = m_objStopTime.Stop();

	//Check the return value of GXGetImage
	GX_VERIFY(emStatus);

	ShowImage(&m_stFrameData);

	UpdateStatisticalData(dElapsedtime);
}

//----------------------------------------------------------------------------------
/**
\brief  Update statistical time data
\param  dData  [in]  Time data

\return void
*/
//----------------------------------------------------------------------------------
void CGxGetImageDlg::UpdateStatisticalData(double dData)
{
	CString strTemp = _T("");

	m_objStatistic.AddStatisticalData(dData);

	// Show the average value 
    strTemp.Format("%.3f",m_objStatistic.GetAverage());
	SetDlgItemText(IDC_EDIT_AVERAGE,strTemp);

	// Show the max value 
	strTemp.Format("%.3f",m_objStatistic.GetMax());
	SetDlgItemText(IDC_EDIT_MAX_VALUE,strTemp);

	// Show the min value
	strTemp.Format("%.3f",m_objStatistic.GetMin());
	SetDlgItemText(IDC_EDIT_MIN_VALUE,strTemp);
}


void CGxGetImageDlg::OnBnClickedBtnCleanTime()
{
	// TODO: Add your control notification handler code here
	SetDlgItemText(IDC_EDIT_TIME_WND,"");
}
