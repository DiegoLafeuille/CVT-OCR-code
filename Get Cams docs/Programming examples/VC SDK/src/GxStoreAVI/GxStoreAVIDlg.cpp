// GxStoreAVIDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxStoreAVI.h"
#include <Windows.h>
#include "GxStoreAVIDlg.h"

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
// CGxStoreAVIDlg dialog

CGxStoreAVIDlg::CGxStoreAVIDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxStoreAVIDlg::IDD, pParent)
	, m_bDevOpened(FALSE)
	, m_bEnableSaveAVI(FALSE)
	, m_hDevice(NULL)
	, m_pBmpInfo(NULL)
	, m_pRawBuffer(NULL)
	, m_pImgBuffer(NULL)
	, m_pAVIFile(NULL)
	, m_nTimeFrame(0)
	, m_psCompressed(NULL)
    , m_ps(NULL)
	, m_hic(NULL)
	, m_bIsColorFilter(false)
	, m_wPlayFPS(15)
	, m_nWrittenDataSize(0)
	, m_pWnd(NULL)
	, m_hDC(NULL)
{
	//{{AFX_DATA_INIT(CGxStoreAVIDlg)
	m_strAVIFolder = _T("");
	m_bCompressedAVI = FALSE;
	m_bShowImage = TRUE;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	memset(&m_Com,0,sizeof(m_Com));

	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CGxStoreAVIDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxStoreAVIDlg)
	DDX_Text(pDX, IDC_EDIT_SAVEAS, m_strAVIFolder);
	DDX_Check(pDX, IDC_CHECK_COMPRESSED_AVI, m_bCompressedAVI);
	DDX_Check(pDX, IDC_CHECK_SHOW_IMAGE, m_bShowImage);
	//}}AFX_DATA_MAP
	DDX_Text(pDX, IDC_EDIT_FPS, m_wPlayFPS);
}

BEGIN_MESSAGE_MAP(CGxStoreAVIDlg, CDialog)
	//{{AFX_MSG_MAP(CGxStoreAVIDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, OnBnClickedBtnOpenDevice)
	ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, OnBnClickedBtnCloseDevice)
	ON_BN_CLICKED(IDC_BTN_SAVE_AS, OnBnClickedBtnSaveAs)
	ON_BN_CLICKED(IDC_BTN_AVI_START, OnBnClickedBtnAviStart)
	ON_BN_CLICKED(IDC_BTN_STOP_AVI, OnBnClickedBtnStopAvi)
	ON_BN_CLICKED(IDC_CHECK_COMPRESSED_AVI, OnBnClickedCheckCompressedAvi)
	ON_BN_CLICKED(IDC_CHECK_SHOW_IMAGE, OnBnClickedCheckShowImage)
	ON_BN_CLICKED(IDC_BTN_SELECT_COMPRESSOR, OnBnClickedBtnSelectCompressor)
	ON_EN_KILLFOCUS(IDC_EDIT_FPS, OnEnKillfocusEditFps)
	//}}AFX_MSG_MAP
	ON_WM_CLOSE()
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxStoreAVIDlg message handlers

BOOL CGxStoreAVIDlg::OnInitDialog()
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

	GX_STATUS emStatus = GX_STATUS_ERROR;

	// Before using any GxIAPI methods, the GxIAPI must be initialized. 
	emStatus = GXInitLib();  
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}

	// Set the path for saving video
	char    szName[MAX_PATH] = {0};
	CString  strAppFullName = ""; 
	GetModuleFileName(NULL, szName, MAX_PATH);
	strAppFullName = szName;
	int nPos = strAppFullName.ReverseFind('\\');
	m_strAVIFolder += strAppFullName.Left(nPos);
    UpdateData(FALSE);

	//If "error C2065: 'CoInitializeEx' : undeclared identifier" appeared in VC6 compiling environment,
	//please add "#define _WIN32_WINNT 0x0400" at the beginning of stdafx.h
	if (CoInitializeEx(NULL, COINIT_MULTITHREADED) != AVIERR_OK)
	{
		return FALSE;
	}

	//Get the pointer and DC handle for showing image
	m_pWnd = GetDlgItem(IDC_STATIC_SHOW_WND);
	m_hDC  = ::GetDC(m_pWnd->m_hWnd);
	
	UpDateUI();
	 
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxStoreAVIDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CGxStoreAVIDlg::OnPaint() 
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

HCURSOR CGxStoreAVIDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

BOOL CGxStoreAVIDlg::PreTranslateMessage(MSG* pMsg)
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
		case IDC_EDIT_FPS:
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

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.
 
\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedBtnOpenDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
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

	GX_OPEN_PARAM stOpenParam;
	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_INDEX;
	stOpenParam.pszContent = "1";

	//Open the first found device 
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	GX_VERIFY(emStatus);
	m_bDevOpened = true;

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

	emStatus = GetDeviceParam();
	GX_VERIFY(emStatus);

	if (!PrepareForShowImg())
	{
		return;
	}

	emStatus = GXRegisterCaptureCallback(m_hDevice, this, OnFrameCallbackFun);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}

	//Set StreamBufferHandlingMode
	emStatus = GXSetEnum(m_hDevice, GX_DS_ENUM_STREAM_BUFFER_HANDLING_MODE, GX_DS_STREAM_BUFFER_HANDLING_MODE_OLDEST_FIRST);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}

	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxStoreAVIDlg::PrepareForShowImg()
{	
	bool bRet = true;

	//Allocate memory for raw image
	m_pRawBuffer = new BYTE[(size_t)m_nPayLoadSize];
	if (m_pRawBuffer == NULL)
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
bool  CGxStoreAVIDlg::PrepareForShowColorImg()
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

	//----------------Compute the frame data for saving AVI file----------
	//Compute the frame size of each image
	m_nFrameSize = m_nImageWidth * m_nImageHeight * 3;

	//---------------------------------------------------------------------
	// Allocate memory for converted RGB image 
	m_pImgBuffer = new BYTE[(size_t)(m_nFrameSize)];
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
bool  CGxStoreAVIDlg::PrepareForShowMonoImg()
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

	//----------------Compute the frame data for saving AVI file----------
	//Compute the frame size of each image
	m_nFrameSize = m_nImageWidth * m_nImageHeight;

	//---------------------------------------------------------------------
	//-------------------Allocate buffer for showing image data------------
	// Allocate memory for showing mono images.
	m_pImgBuffer = new BYTE[(size_t)(m_nFrameSize)];
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
void CGxStoreAVIDlg::UnPrepareForShowImg()
{
	if (m_pRawBuffer != NULL)
	{
		delete []m_pRawBuffer;
		m_pRawBuffer = NULL;
	}

	if (m_pImgBuffer != NULL)
	{
		delete []m_pImgBuffer;
		m_pImgBuffer = NULL;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This is a acquisition callback function that demonstrates how to process the acquired image
\param   pFrame 

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxStoreAVIDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxStoreAVIDlg *pDlg  = (CGxStoreAVIDlg*)(pFrame->pUserParam);
	VxUint32 nImageHeight = (VxUint32)pDlg->m_nImageHeight;
	VxUint32 nImageWidth  = (VxUint32)pDlg->m_nImageWidth;

	if (pFrame->status == 0)
	{
		memcpy(pDlg->m_pRawBuffer,pFrame->pImgBuf,pFrame->nImgSize);

		if (pDlg->m_bIsColorFilter)
		{
			// If the acquired image is color format,convert it to RGB 
			DxRaw8toRGB24(pDlg->m_pRawBuffer, pDlg->m_pImgBuffer, nImageWidth , nImageHeight, RAW2RGB_NEIGHBOUR,
				DX_PIXEL_COLOR_FILTER(pDlg->m_nPixelColorFilter), TRUE);
		}
		else
		{
			// If the acquired image is mono format,you must flip the image data for showing.
			for(VxUint32 i = 0;i <nImageHeight;i++)
			{
				memcpy(pDlg->m_pImgBuffer+i*nImageWidth, pDlg->m_pRawBuffer+(nImageHeight-i-1)*nImageWidth,(size_t)nImageWidth);
			}
		}		

		pDlg->DrawImg(pDlg->m_pImgBuffer);

		// Save AVI
		pDlg->m_SaverCriSec.Enter();
		pDlg->ExSaveAVI(pDlg->m_pImgBuffer);

		//If you don't need to check the disk space or limit the size of AVI file, you could use the following code to save the AVI file
		//pDlg->SaveAVI(pDlg->m_pImgBuffer); 

		pDlg->m_SaverCriSec.Leave();
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to draw image in the window
\param   pImageBuffer [in]   

\return  void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::DrawImg(BYTE *pImageBuffer)
{   
	if (!m_bShowImage)
	{
		return;
	}

	int  nWndWidth  = 0;
	int  nWndHeight = 0;

	// Initialize the parameters for drawing image.
	RECT objRect;
	m_pWnd->GetClientRect(&objRect);	
	nWndWidth  = objRect.right  - objRect.left;
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
		           (int)m_nImageWidth,
		           (int)m_nImageHeight,
		            pImageBuffer,
		            m_pBmpInfo,
		            DIB_RGB_COLORS,
		            SRCCOPY
		            );	
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to get the parameters of the device

\return GX_STATUS_SUCCESS: Success Else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxStoreAVIDlg::GetDeviceParam()
{
	GX_STATUS emStatus       = GX_STATUS_ERROR;
	bool      bIsImplemented = false;

	// Get image size
	emStatus = GXGetInt(m_hDevice, GX_INT_PAYLOAD_SIZE, &m_nPayLoadSize);
	VERIFY_STATUS_RET(emStatus);

	// Get image width
	emStatus = GXGetInt(m_hDevice, GX_INT_WIDTH, &m_nImageWidth);
	VERIFY_STATUS_RET(emStatus);

	// Get image height
	emStatus = GXGetInt(m_hDevice, GX_INT_HEIGHT, &m_nImageHeight);
	VERIFY_STATUS_RET(emStatus);

	// Check whether the device supports PixelColorFilter
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_bIsColorFilter);
	VERIFY_STATUS_RET(emStatus);

	if(m_bIsColorFilter)
	{
		emStatus = GXGetEnum(m_hDevice, GX_ENUM_PIXEL_COLOR_FILTER, &m_nPixelColorFilter);
	}

	return emStatus;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return  void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxStoreAVIDlg::InitDevice()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerMode off
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);	
	VERIFY_STATUS_RET(emStatus);

	// If the device support the pixel format of"GX_PIXEL_FORMAT_BAYER_GR8",
	// the following code can be used to set the pixel format to GX_PIXEL_FORMAT_BAYER_GR8:
	// emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_GR8);
	// VERIFY_STATUS_RET(emStatus);

	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8bit(); 

	return emStatus;
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxStoreAVIDlg::SetPixelFormat8bit()
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
		if (pEnumDescription != NULL)
		{
			delete []pEnumDescription;
			pEnumDescription = NULL;
		}
	}

	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close device.
 
\return void 
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedBtnCloseDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_ERROR;

	// If the device is acquiring image then stop it.
    if (m_bEnableSaveAVI)
    {
		m_bEnableSaveAVI = FALSE;
		m_SaverCriSec.Enter();
		UnPrepareForSaveAVI();
		m_SaverCriSec.Leave();
    }

	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}

	//Unregister the CaptureCallback function 
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}

	// Release memory for showing image
	UnPrepareForShowImg();
    
	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}

	m_hDevice    = NULL;
	m_bDevOpened = FALSE;

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to select the saving path for AVI file

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedBtnSaveAs() 
{
	// TODO: Add your control notification handler code here
	char arrFilePath[MAX_PATH*2] = {0};

	BROWSEINFO br;
	br.hwndOwner = GetSafeHwnd();
	br.iImage = 0;
	br.pszDisplayName = 0;
	br.lParam = 0 ;
	br.lpfn = 0 ;
	br.lpszTitle = "Please select the saving path for stream ";
	br.pidlRoot = 0 ;
	br.ulFlags = BIF_RETURNONLYFSDIRS;
	if ( SHGetPathFromIDList(SHBrowseForFolder(&br),arrFilePath) != TRUE)
	{
		return ;
	}

	m_strAVIFolder = arrFilePath; 

	UpdateData(FALSE);	
}


//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to start recording 

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedBtnAviStart() 
{
	// TODO: Add your control notification handler code here
	BOOL bRet = FALSE;

	// Allocate memory for saving AVI file 
	bRet = PrepareForSaveAVI();
	if (!bRet)
	{
		UnPrepareForSaveAVI();
		MessageBox("Fail to start recording! ");
		return;
	}

	m_bEnableSaveAVI = TRUE;
	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop recording

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedBtnStopAvi() 
{
	// TODO: Add your control notification handler code here
	m_bEnableSaveAVI = FALSE;
	m_SaverCriSec.Enter();
	UnPrepareForSaveAVI();
    m_SaverCriSec.Leave();
	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to send the "Show image" command
 
\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedCheckShowImage() 
{
	// TODO: Add your control notification handler code here
	UpdateData(TRUE);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to send the "Compress Video" command

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedCheckCompressedAvi() 
{
	// TODO: Add your control notification handler code here
    UpdateData(TRUE);

    UpDateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to update UI

\return  void
*/
//----------------------------------------------------------------------------------
void  CGxStoreAVIDlg::UpDateUI()
{
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bDevOpened);
	GetDlgItem(IDC_CHECK_SHOW_IMAGE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_BTN_SAVE_AS)->EnableWindow(m_bDevOpened && !m_bEnableSaveAVI);
	GetDlgItem(IDC_CHECK_COMPRESSED_AVI)->EnableWindow(m_bDevOpened && !m_bEnableSaveAVI);
	GetDlgItem(IDC_BTN_STOP_AVI)->EnableWindow(m_bDevOpened && m_bEnableSaveAVI);
	GetDlgItem(IDC_BTN_SELECT_COMPRESSOR)->EnableWindow(m_bDevOpened && m_bCompressedAVI && !m_bEnableSaveAVI);
	GetDlgItem(IDC_BTN_AVI_START)->EnableWindow(m_bDevOpened && !m_bEnableSaveAVI);
	GetDlgItem(IDC_EDIT_FPS)->EnableWindow(m_bDevOpened && !m_bEnableSaveAVI);
}
//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in] 

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::ShowErrorString(GX_STATUS emErrorStatus)
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
		MessageBox("Fail to call the GXGetLastError interface! ");
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
/**
\brief  This function demonstrates how to close the application.

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnClose()
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_ERROR;

	// If the device is acquiring image then stop it.
	if (m_bEnableSaveAVI)
	{
		m_SaverCriSec.Enter();
		m_bEnableSaveAVI = FALSE;
		UnPrepareForSaveAVI();
		m_SaverCriSec.Leave();
		ResetCompressor();
	}

	if (m_bDevOpened)
	{
		// Send AcquisitionStop commands
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);

		//Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);

		// Release memory for showing image 
		UnPrepareForShowImg();

		emStatus = GXCloseDevice(m_hDevice);
	}

	::ReleaseDC(m_pWnd->m_hWnd, m_hDC);

	//Release GxIAPI
	emStatus = GXCloseLib();

	CoUninitialize();
	CDialog::OnClose();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for saving AVI file

\return TRUE:Success  FALSE:Fail
*/
//----------------------------------------------------------------------------------
BOOL CGxStoreAVIDlg::PrepareForSaveAVI()
{
	BOOL    bRet           = FALSE;
	LONG    cbFormat       = 0;    

	if (m_bIsColorFilter)
	{
		cbFormat = sizeof(BITMAPINFOHEADER);
	}
	else
	{
		cbFormat = sizeof(BITMAPINFOHEADER) + (256 * 4);
	}

	AVISTREAMINFO StreamInfo;
	HRESULT hr;

	SYSTEMTIME system;
	GetLocalTime(&system);
	string strAVIFileName = MsgComposer("\\%d_%d_%d_%d_%d_%d_%d.avi",system.wYear,
		                                                             system.wMonth,
																     system.wDay,
		                                                             system.wHour, 
		                                                             system.wMinute, 
		                                                             system.wSecond, 
		                                                             system.wMilliseconds);

	string strAVIFilePath = m_strAVIFolder;
	strAVIFilePath = strAVIFilePath + strAVIFileName;
	::DeleteFile(strAVIFilePath.c_str());

	AVIFileInit();	
	//Create a AVI file 
	hr = AVIFileOpen(&m_pAVIFile, strAVIFilePath.c_str(), OF_WRITE | OF_CREATE, NULL); 
	if (hr != AVIERR_OK)
	{
		return FALSE;
	}

	// Create a stream
	::ZeroMemory(&StreamInfo, sizeof(StreamInfo));
	StreamInfo.fccType                = streamtypeVIDEO;
	StreamInfo.fccHandler             = 0;
	StreamInfo.dwScale                = 1;                                       
	StreamInfo.dwRate                 = m_wPlayFPS;  
	StreamInfo.dwSuggestedBufferSize  = DWORD(m_nImageWidth * m_nImageHeight * 3); 
	SetRect(&StreamInfo.rcFrame, 0, 0, (int)m_nImageWidth, (int)m_nImageHeight);
	hr = AVIFileCreateStream(m_pAVIFile, &m_ps, &StreamInfo);
	if (hr != AVIERR_OK)
	{
		return FALSE;
	}

	//Initialize the structure for compressing stream
	AVICOMPRESSOPTIONS opts;
	AVICOMPRESSOPTIONS FAR * aopts[1] = {&opts};

	if (m_bCompressedAVI)
	{
		if (m_hic == NULL)
		{
			return FALSE;
		}

		DWORD dwSize = ICGetStateSize(m_hic);
		LPVOID lpvod = (LPVOID)GlobalAlloc(NULL,dwSize);

		ICINFO icinfo;
		ICGetInfo(m_hic, &icinfo, sizeof(icinfo)); 
		opts.fccType           = icinfo.fccType;
		opts.fccHandler        = icinfo.fccHandler;
		opts.dwKeyFrameEvery   = 0;
		opts.dwQuality         = 5000;
		opts.dwBytesPerSecond  = 0;
		opts.dwFlags           = 8;
		opts.lpFormat          = 0;
		opts.cbFormat          = 0;
		opts.lpParms           = lpvod;
		opts.cbParms           = dwSize;
		opts.dwInterleaveEvery = 0;
	}
    else
	{
		opts.fccHandler = mmioFOURCC('D', 'I', 'B', ' '); // 0x20424944 DIB 
	}

	//Create compressed stream
	hr = AVIMakeCompressedStream(&m_psCompressed, m_ps, &opts, NULL);
	if (hr != AVIERR_OK)
	{
		return FALSE;
	}

	//Set the format of the stream file that user input
	hr = AVIStreamSetFormat(m_psCompressed, 0, &m_pBmpInfo->bmiHeader, cbFormat);
	if (hr != AVIERR_OK)
	{
		return FALSE;
	}
	else
	{
		bRet = TRUE;
	}

	return bRet;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release memory for creating AVI file and stream

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::UnPrepareForSaveAVI()
{
	if (m_psCompressed != NULL)
	{
		AVIStreamClose(m_psCompressed);
		m_psCompressed = NULL;
	}
	if (m_ps != NULL)
	{
		AVIStreamClose(m_ps);
		
		m_ps = NULL;
	}
	if (m_pAVIFile != NULL)
	{
		AVIFileEndRecord(m_pAVIFile);
		AVIFileClose(m_pAVIFile);
		m_pAVIFile = NULL;
	}

	AVIFileExit();
    m_nTimeFrame = 0;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release memory for video compressor
 
\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::ResetCompressor()
{
	if (m_hic != NULL)
	{
		ICClose(m_hic);
		m_hic = NULL;
	}

	ICCompressorFree(&m_Com);
	memset(&m_Com,0,sizeof(m_Com));
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to save AVI file
\param  pImageBuf  [in]  

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::ExSaveAVI(BYTE * pImageBuf)
{
	if (!m_bEnableSaveAVI)
	{
		return;
	}

	LONG nBufferSize = 0; 
	BOOL bRet        = TRUE;

	// ---------------------------------------------
	// Check the free space of current directory, stop saving if the free space is not enough
	if (GetDiskFreeSpace() < m_nFrameSize + AVI_FRAME_OFFSET)
	{
		m_bEnableSaveAVI = FALSE;
		m_SaverCriSec.Enter();
		UnPrepareForSaveAVI();
		m_SaverCriSec.Leave();
		UpDateUI();
		return;
	}

	// Check whether the file size would reach to the max value,
	// if so then create a new AVI file to continue to record
	if ((m_nFrameSize + m_nWrittenDataSize + AVI_FRAME_OFFSET) > AVI_MAX_SIZE - AVI_HEADERSIZE)
    {
		bRet = ReCreatVedioFile();
		if (!bRet)
		{
			return;
		}

		m_nWrittenDataSize = 0;
    }

	HRESULT hr = AVIStreamWrite(m_psCompressed,
		                        m_nTimeFrame,
		                        1, 
		                        pImageBuf,	
		                        (LONG)m_nFrameSize, 
		                        AVIIF_KEYFRAME, 
		                        NULL,
		                        &nBufferSize);
	
	if (hr == AVIERR_OK )
	{
		m_nTimeFrame ++;

		// Compute the data size of AVI file 
		m_nWrittenDataSize = m_nWrittenDataSize + nBufferSize + AVI_FRAME_OFFSET;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to save AVI file
\param  pImageBuf  [in]  
\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::SaveAVI(BYTE * pImageBuf)
{
	if (!m_bEnableSaveAVI)
	{
		return;
	}

	HRESULT hr = AVIStreamWrite(m_psCompressed,
		                        m_nTimeFrame,
		                        1, 
		                        pImageBuf,	
		                       (LONG)m_nFrameSize, 
		                        AVIIF_KEYFRAME, 
		                        NULL,
		                        NULL);

	// Check whether the device has written data successfully 
	if (hr == AVIERR_OK )
	{
		m_nTimeFrame ++;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop recording and create a new AVI file to record again

\return TRUE:Success FALSE:Fail
*/
//----------------------------------------------------------------------------------
BOOL CGxStoreAVIDlg::ReCreatVedioFile()
{
	BOOL bRet = TRUE;

	// Stop saving
	m_SaverCriSec.Enter();
	UnPrepareForSaveAVI();
	m_SaverCriSec.Leave();

    // Create a new AVI file
	bRet = PrepareForSaveAVI();

	return bRet;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to select the video compressor

\return void
*/
//----------------------------------------------------------------------------------
void CGxStoreAVIDlg::OnBnClickedBtnSelectCompressor()
{
	// TODO: Add your control notification handler code here 
	LPBITMAPINFOHEADER lpbi = &m_pBmpInfo->bmiHeader;
	DWORD  fccType = ICTYPE_VIDEO;

	m_Com.cbSize=sizeof(m_Com); 

	//You could set the compressor by changing the parameters of mmioFOURCC,for example,
	//HIC hic = ICOpen(ICTYPE_VIDEO,mmioFOURCC('d','i','v','x'),ICMODE_COMPRESS);

	//If you don't know the type and the name of the compressor,following code could be used:
	BOOL bRet = ICCompressorChoose(m_hWnd,ICMF_CHOOSE_ALLCOMPRESSORS ,NULL,NULL,&m_Com,"Select the type of compression");
	if (!bRet)
	{
		// The bRet is false means user hasn't selected the compressor
		return ;
	}
	HIC hic =ICOpen(m_Com.fccType, m_Com.fccHandler, ICMODE_QUERY); 

	// If you have selected the encode then check whether the compressor is available
	if (hic != NULL) 
	{ 
		if (fccType == ICTYPE_VIDEO && ICCompressQuery(hic, lpbi, NULL) != ICERR_OK)
		{
			ResetCompressor();
			MessageBox("The compressor you selected is not available,please select again! ");
			return ;
		}

		m_hic = hic;
	}
}


void CGxStoreAVIDlg::OnEnKillfocusEditFps()
{
	// TODO: Add your control notification handler code here
	UpdateData(TRUE);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to get the free space of the saving path

\return nFreeBytes  
*/
//----------------------------------------------------------------------------------
int64_t CGxStoreAVIDlg::GetDiskFreeSpace()  
{       
	int64_t nFreeBytes = 0;        
    BOOL    bResult;                         
	
    bResult = GetDiskFreeSpaceEx (m_strAVIFolder, 0, 0,             
							     (PULARGE_INTEGER)&nFreeBytes);                                   

    return nFreeBytes;      
}      
