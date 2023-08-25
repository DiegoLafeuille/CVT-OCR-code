// GxSnapRaw16Dlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxSnapRaw16.h"
#include "GxSnapRaw16Dlg.h"

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
// CGxSnapRaw16Dlg dialog

CGxSnapRaw16Dlg::CGxSnapRaw16Dlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxSnapRaw16Dlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CGxSnapRaw16Dlg)
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_strSavePath           = "";              // The path for saving image 
	m_hDevice               = NULL;            // The handle for device  
	m_pBmpInfo              = NULL;	           // The pointer for showing BMP image 
	m_pRawBuf               = NULL;            // The pointer of the raw image buffer
	m_pRaw8Buf              = NULL;            // The pointer of 8Bit raw image buffer 
	m_pShowBuf              = NULL;            // The pointer for showing image 
	m_nImageWidth           = 0;               // Acquired image width
	m_nImageHeight          = 0;               // Acquired image height
	m_nPayLoadSize          = 0;               // Acquired raw image size 
	m_nPixelColorFilter     = 0;               // PixelColorFilter
	m_bIsColorFilter        = false;           // The flag indicates whether the device supports ColorFilter
	m_bIsOpen               = false;           // The flag indicates whether the first device has been opened
	m_bIsSnap               = false;           // The flag indicates whether AcquisitionStart command has been send
	m_bIsSaveImg            = false;           // The flag indicates whether the image is saved
    m_hWndHandle            = NULL;            // The handle for showing image window  
    m_hDC                   = NULL;            // The handle for drawing image DC 

	memset(m_chBmpBuf,0,sizeof(m_chBmpBuf));
}

void CGxSnapRaw16Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxSnapRaw16Dlg)
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxSnapRaw16Dlg, CDialog)
	//{{AFX_MSG_MAP(CGxSnapRaw16Dlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_OPEN_DEVICE, OnOpenDevice)
	ON_BN_CLICKED(IDC_START_SNAP, OnStartSnap)
	ON_BN_CLICKED(IDC_STOP_SNAP, OnStopSnap)
	ON_BN_CLICKED(IDC_CLOSE_DEVICE, OnCloseDevice)
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_SAVE_RAW_IMAGE, OnSaveRawImage)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxSnapRaw16Dlg message handlers

BOOL CGxSnapRaw16Dlg::OnInitDialog()
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
	int     nPos = 0;
	char    chPathName[MAX_PATH] = {0};   
	CString strPathName = ""; 

	// Before using any GxIAPI methods, the GxIAPI must be initialized.
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	emStatus = GXInitLib();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		exit(0);
	}

	// Set the path for showing image 
	GetModuleFileName(NULL, chPathName, MAX_PATH);
	strPathName = chPathName;
	nPos = strPathName.ReverseFind('\\');
	m_strSavePath += strPathName.Left(nPos);
	m_strSavePath += "\\GxSnapRaw16Images";

	// Get the pointer and DC handle for showing image
	GetDlgItem(IDC_STATIC_SHOW_IMAGE, &m_hWndHandle);
    m_hDC  = ::GetDC(m_hWndHandle); 
	::GetClientRect(m_hWndHandle, &m_objRect);	     
	
	UpdateUI();

	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxSnapRaw16Dlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CGxSnapRaw16Dlg::OnPaint() 
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
HCURSOR CGxSnapRaw16Dlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close the application.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::OnClose() 
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
		m_hDevice = NULL;
	}
	
	// Release GxIAPI
	emStatus = GXCloseLib();
	
	::ReleaseDC(m_hWndHandle, m_hDC);

	CDialog::OnClose();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to enumerate all camera devices and initialize device.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::OnOpenDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS      emStatus     = GX_STATUS_SUCCESS;
	uint32_t       nDeviceNum   = 0;
	int64_t        nPixelFormat = 0;        
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

	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to start acquisition

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::OnStartSnap() 
{
	// TODO: Add your control notification handler code here
	bool      bReturn  = false;
    GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	bReturn  = PrepareForShowImage();
	if (!bReturn)
	{
		MessageBox("Fail to allocate resource for showing image!");
		return;
	}
	
	//Register the CaptureCallback function 
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

	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to stop acquisition

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::OnStopSnap() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// Send AcquisitionStop Command
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);
	
	//Unregister the CaptureCallback function
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	GX_VERIFY(emStatus);
    
	// Release memory for showing image
	UnPrepareForShowImage();
	m_bIsSnap = false;

	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close device.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::OnCloseDevice() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	
	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
		// Send AcquisitionStop command
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

	UpdateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the switch for saving the BMP images
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::OnSaveRawImage() 
{
	// TODO: Add your control notification handler code here
	m_bIsSaveImg = !m_bIsSaveImg;	
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return  emStatus GX_STATUS_SUCCESS:Success, else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxSnapRaw16Dlg::InitParam()
{
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	CString   strSymbolic = "";
	
	// Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// Set the TriggerMode off
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);	
	VERIFY_STATUS_RET(emStatus);
	
	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to non-8-bit.
	emStatus = SetPixelSize16(&strSymbolic);
	VERIFY_STATUS_RET(emStatus);

	// Show the pixel format to UI
	GetDlgItem(IDC_EDIT_PIXEL_FORMAT)->SetWindowText(strSymbolic);
	
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

//----------------------------------------------------------------------------------
/**
\brief    This function demonstrates how to set the pixel format to 16bit 
\param    pstrSymbolic  

\return   emStatus GX_STATUS_SUCCESS:Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxSnapRaw16Dlg::SetPixelSize16(CString  *pstrSymbolic)
{
	uint32_t  i           = 0;                   
	uint32_t  nEnmuEntry  = 0;
	size_t    nBufferSize = 0;	
	GX_STATUS emStatus    = GX_STATUS_SUCCESS;
	GX_ENUM_DESCRIPTION  *pEnumDescription = NULL;
	GX_ENUM_DESCRIPTION  *pEnumTemp        = NULL;
	
	// Get the enumeration entry of the pixel format the device supports.
	emStatus = GXGetEnumEntryNums(m_hDevice, GX_ENUM_PIXEL_FORMAT, &nEnmuEntry);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		*pstrSymbolic = "\0";
		return emStatus;
	}
		
	// Allocate memory for getting the enumeration entry of the pixel format.
	nBufferSize      = nEnmuEntry * sizeof(GX_ENUM_DESCRIPTION);
	pEnumDescription = new GX_ENUM_DESCRIPTION[nEnmuEntry];

	emStatus = GXGetEnumDescription(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumDescription, &nBufferSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		*pstrSymbolic = "\0";

		// Release memory
		if (pEnumDescription != NULL)
		{
			delete []pEnumDescription;
			pEnumDescription = NULL;
		}
		return emStatus;
	}
			
	// A loop to visit every enumeration entry node once to find which enumeration entry node is 16bit and then set pixel format to its value 
	pEnumTemp = pEnumDescription;
	for (i = 0; i < nEnmuEntry; i++)
	{
		if ((pEnumTemp->nValue & GX_PIXEL_16BIT) == GX_PIXEL_16BIT)
		{
			emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, pEnumTemp->nValue);
			*pstrSymbolic = pEnumTemp->szSymbolic;
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

\return  true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool CGxSnapRaw16Dlg::PrepareForShowImage()
{	
	int i = 0;
	size_t nImageSize = 0;

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
	
	//Allocate memory for showing raw images.
	m_pRawBuf = new BYTE[(size_t)(m_nPayLoadSize)];
	if (m_pRawBuf == NULL)
	{
		return false;
	}

	// Allocate memory for showing 8bit images.
	m_pRaw8Buf = new BYTE[(size_t)(m_nImageHeight * m_nImageWidth)];
	if (m_pRaw8Buf == NULL)
	{
		if (m_pRawBuf != NULL )
		{
			delete []m_pRawBuf;
			m_pRawBuf = NULL;
		}
		return false;
	}

	// The size of the image 
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
		if (m_pRaw8Buf != NULL )
		{
			delete []m_pRaw8Buf;
			m_pRaw8Buf = NULL;
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
void CGxSnapRaw16Dlg::UnPrepareForShowImage()
{
	if (m_pRawBuf != NULL)
	{
		delete []m_pRawBuf;
		m_pRawBuf = NULL;
	}

	if (m_pRaw8Buf != NULL)
	{
		delete []m_pRaw8Buf;
		m_pRaw8Buf = NULL;
	}
		
	if (m_pShowBuf != NULL)
	{
		delete []m_pShowBuf;
		m_pShowBuf = NULL;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This is a acquisition callback function that demonstrates how to process the acquired image
\param   pFrame                           

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxSnapRaw16Dlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM *pFrame)
{
	CGxSnapRaw16Dlg *pDlg   = (CGxSnapRaw16Dlg*)(pFrame->pUserParam);
	VxUint32   i       = 0;
	VxUint32   nHeight = (VxUint32)pDlg->m_nImageHeight;
	VxUint32   nWidth  = (VxUint32)pDlg->m_nImageWidth;

	if (pFrame->status != 0)
	{
		return;
	}
    
	// Copy the image data from callback function
	memcpy(pDlg->m_pRawBuf, pFrame->pImgBuf, pFrame->nImgSize);

	if (pDlg->m_bIsSaveImg)
	{
		pDlg->SaveRawImage(pDlg->m_pRawBuf, (nWidth * nHeight * 2), pDlg->m_strSavePath);
	}

	// Covert the raw16 image to the raw8 image for showing 
	DxRaw16toRaw8(pDlg->m_pRawBuf, pDlg->m_pRaw8Buf, nWidth, nHeight, DX_BIT_0_7);
	
	if (pDlg->m_bIsColorFilter)
	{
		// Convert the raw image to 24bit RGB image for showing 
		DxRaw8toRGB24(pDlg->m_pRaw8Buf, pDlg->m_pShowBuf, nWidth, nHeight, RAW2RGB_NEIGHBOUR, 
			          DX_PIXEL_COLOR_FILTER(pDlg->m_nPixelColorFilter), TRUE);
	}
	else
	{
		// If the acquired image is mono format,you must flip the image data for showing.
		for(i = 0; i < nHeight; i++)
		{
			memcpy(pDlg->m_pShowBuf + i * nWidth, pDlg->m_pRaw8Buf + (nHeight - i - 1) * nWidth, nWidth);
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
\param   pWnd        

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::DrawImage(BYTE *pImageBuf, int nWidth, int nHeight, BITMAPINFO *pBmpInfo)
{
	if ((pImageBuf == NULL) || (pBmpInfo == NULL))
	{
		return;
	}

	int nWndWidth  = m_objRect.right - m_objRect.left;       
	int nWndHeight = m_objRect.bottom - m_objRect.top;     

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

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to save the raw image 
\param   pRawBuf      
\param   nImageSize   
\param   srtSavePath  

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::SaveRawImage(BYTE *pRawBuf, int nImageSize, CString srtSavePath)
{
	if ((pRawBuf == NULL) || (srtSavePath == ""))
	{
		return;
	}

	CString strFileName = "";               
	DWORD   dwImageSize = nImageSize;     
	DWORD   dwBytesRead = 0;               

	// Get current time
	SYSTEMTIME  sysTime;
	GetLocalTime(&sysTime);

	//Create the directory for showing image 
	BOOL bRet = ::CreateDirectory(m_strSavePath,NULL);   
	if(bRet)   
	{   
		::SetFileAttributes(m_strSavePath,FILE_ATTRIBUTE_NORMAL);   
    } 

	// Get current time as the filename of images
	strFileName.Format("%s\\%d_%d_%d_%d_%d_%d_%d.raw",
		srtSavePath,
		sysTime.wYear,
		sysTime.wMonth,
		sysTime.wDay,
		sysTime.wHour,
		sysTime.wMinute,
		sysTime.wSecond,
		sysTime.wMilliseconds);
	   
	HANDLE hFile = ::CreateFile(strFileName,
		GENERIC_WRITE,
		FILE_SHARE_READ,
		NULL,
		CREATE_ALWAYS,														
		FILE_ATTRIBUTE_NORMAL,
		NULL);
	
	if (hFile == INVALID_HANDLE_VALUE)   
	{
		return;
	}
	else                               
	{ 
		::WriteFile(hFile, pRawBuf, dwImageSize, &dwBytesRead, NULL);
		CloseHandle(hFile);
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus 

 \return void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::ShowErrorString(GX_STATUS emErrorStatus)
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

//----------------------------------------------------------------------------------
/**
\brief   Update UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSnapRaw16Dlg::UpdateUI()
{
	GetDlgItem(IDC_OPEN_DEVICE)->EnableWindow(!m_bIsOpen);
	GetDlgItem(IDC_START_SNAP)->EnableWindow(m_bIsOpen && (!m_bIsSnap));
	GetDlgItem(IDC_STOP_SNAP)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);
}



BOOL CGxSnapRaw16Dlg::PreTranslateMessage(MSG* pMsg) 
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
