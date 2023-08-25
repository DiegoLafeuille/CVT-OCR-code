// GxSingleCamMonoDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxSingleCamMono.h"
#include "GxSingleCamMonoDlg.h"

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
// CGxSingleCamMonoDlg dialog

CGxSingleCamMonoDlg::CGxSingleCamMonoDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxSingleCamMonoDlg::IDD, pParent)
	, m_bIsSaveImg(FALSE)
	, m_bDevOpened(FALSE)
	, m_bIsSnap(FALSE)
	, m_bTriggerMode(false)
	, m_bTriggerActive(false)
	, m_bTriggerSource(false)
	, m_nImageHeight(0)
	, m_nImageWidth(0)
	, m_nPayLoadSize(0)
	, m_nTriggerMode(0)
	, m_hDevice(NULL)
	, m_pBmpInfo(NULL)
	, m_pBufferRaw(NULL)
	, m_pImageBuffer(NULL)
	, m_strFilePath("")
	, m_pWnd(NULL)
	, m_hDC(NULL)
{
	//{{AFX_DATA_INIT(CGxSingleCamMonoDlg)
	m_dShutterValue = 0.0;
	m_dGainValue = 0.0;
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	memset(&m_stGainFloatRange,0,sizeof(GX_FLOAT_RANGE));
	memset(&m_stShutterFloatRange,0,sizeof(GX_FLOAT_RANGE));
	memset(m_chBmpBuf,0,sizeof(m_chBmpBuf));
}

void CGxSingleCamMonoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxSingleCamMonoDlg)
	DDX_Text(pDX, IDC_EDIT_SHUTTER, m_dShutterValue);
	DDX_Text(pDX, IDC_EDIT_GAIN, m_dGainValue);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxSingleCamMonoDlg, CDialog)
	//{{AFX_MSG_MAP(CGxSingleCamMonoDlg)
	ON_WM_QUERYDRAGICON()
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_CLOSE()
	ON_BN_CLICKED(IDC_BTN_SOFTTRIGGER, OnBnClickedBtnSofttrigger)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_MODE, OnCbnSelchangeComboTriggerMode)
	ON_BN_CLICKED(IDC_BTN_OPEN_DEVICE, OnBnClickedBtnOpenDevice)
	ON_BN_CLICKED(IDC_BTN_CLOSE_DEVICE, OnBnClickedBtnCloseDevice)
	ON_BN_CLICKED(IDC_BTN_STOP_SNAP, OnBnClickedBtnStopSnap)
	ON_BN_CLICKED(IDC_CHECK_SAVE_IMAGE, OnBnClickedCheckSaveImage)
	ON_BN_CLICKED(IDC_BTN_START_SNAP, OnBnClickedBtnStartSnap)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_ACTIVE, OnCbnSelchangeComboTriggerActive)
	ON_CBN_SELCHANGE(IDC_COMBO_TRIGGER_SOURCE, OnCbnSelchangeComboTriggerSource)
	ON_EN_KILLFOCUS(IDC_EDIT_SHUTTER, OnKillfocusEditShutter)
	ON_EN_KILLFOCUS(IDC_EDIT_GAIN, OnKillfocusEditGain)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxSingleCamMonoDlg message handlers

BOOL CGxSingleCamMonoDlg::OnInitDialog()
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

	// Set the path for saving image 
	CString strDate = "";
	char    szName[MAX_PATH] = {0};
	CString strAppFullName = ""; 
	GetModuleFileName(NULL, szName, MAX_PATH);
	strAppFullName = szName;
	int nPos = strAppFullName.ReverseFind('\\');
	m_strFilePath += strAppFullName.Left(nPos);
	m_strFilePath += "\\GxSingleCamMonoImages";

	//Get the pointer and DC handle for showing image
	m_pWnd = GetDlgItem(IDC_STATIC_SHOW_FRAME);
	m_hDC  = ::GetDC(m_pWnd->m_hWnd);
 
	UpDateUI();
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxSingleCamMonoDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char chFileName[MAX_PATH] = {'\0'};
		GetModuleFileName(NULL, chFileName, MAX_PATH);
		CFileVersion fv(chFileName);
		CAboutDlg objDlgAbout;		
		objDlgAbout.m_strProductVersion = _T("Version: ") + fv.GetProductVersion();
		objDlgAbout.m_strLegalCopyRight = fv.GetLegalCopyright();
		objDlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CGxSingleCamMonoDlg::OnPaint() 
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
HCURSOR CGxSingleCamMonoDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to close the application.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnClose() 
{
	// TODO: Add your message handler code here and/or call default

	GX_STATUS emStatus = GX_STATUS_SUCCESS;
    
	// If the device is acquiring image then stop it.
    if (m_bIsSnap)
    {
		// Send AcquisitionStop command
		emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);

		// Unregister the CaptureCallback function 
		emStatus = GXUnregisterCaptureCallback(m_hDevice);

		m_bIsSnap = FALSE;

		// Release memory for showing image
		UnPrepareForShowImg();
    }


	::ReleaseDC(m_pWnd->m_hWnd, m_hDC);

	// If the device is opened then close it.
	if (m_bDevOpened)
	{
		emStatus = GXCloseDevice(m_hDevice);
		m_bDevOpened = FALSE;
		m_hDevice    = NULL;
	}
	
	// Release GxIAPI
	emStatus = GXCloseLib();

	CDialog::OnClose();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the TriggerMode Combobox 
         
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::InitTriggerModeUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports TriggerMode
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_TRIGGER_MODE, &m_bTriggerMode);
	GX_VERIFY(emStatus);
	if (!m_bTriggerMode)
	{
		return;
	}

	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
	GX_VERIFY(emStatus);

	// Initialize the TriggerMode Combobox 
	InitEnumUI(GX_ENUM_TRIGGER_MODE, (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_MODE), m_bTriggerMode);
}

//---------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize the TriggerSource Combobox 

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::InitTriggerSourceUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports TriggerSource
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_TRIGGER_SOURCE, &m_bTriggerSource);
	GX_VERIFY(emStatus);
	if (!m_bTriggerSource)
	{
		return;
	}

	// Initialize the TriggerSource Combobox
	InitEnumUI(GX_ENUM_TRIGGER_SOURCE, (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE),m_bTriggerSource);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize TriggerActivation Combobox 

\return  void  
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::InitTriggerActivationUI()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Check whether the device supports TriggerActivation
	emStatus = GXIsImplemented(m_hDevice, GX_ENUM_TRIGGER_ACTIVATION, &m_bTriggerActive);
	GX_VERIFY(emStatus);
	if (!m_bTriggerActive)
	{
		return;
	}

	// Initialize the TriggerActivation Combobox
	InitEnumUI(GX_ENUM_TRIGGER_ACTIVATION, (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE), m_bTriggerActive);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::InitUI()
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	int  nValue = 0;
	
	InitTriggerModeUI();

	InitTriggerSourceUI();

	InitTriggerActivationUI();

	InitShutterUI();

	InitGainUI();

	UpdateData(false);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the ExposureTime control
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::InitShutterUI()
{
	CStatic     *pStcShutterShow = (CStatic *)GetDlgItem(IDC_STC_SHUTTER_SHOW);
    CEdit       *pEditShutterVal = (CEdit *)GetDlgItem(IDC_EDIT_SHUTTER);
    
	// Check whether the edit control is available
	if ((pEditShutterVal == NULL) || (pStcShutterShow == NULL))
	{
		return;
	}

    GX_STATUS      emStatus = GX_STATUS_ERROR;
	CString        strTemp  = "";
	double  dShutterValue   = 0.0;

	// Get the range of ExposureTime value
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &m_stShutterFloatRange);
	GX_VERIFY(emStatus);

	//Show the range in the static control
    strTemp.Format("Exposure(%.4f~%.4f)%s", m_stShutterFloatRange.dMin, m_stShutterFloatRange.dMax, m_stShutterFloatRange.szUnit);
	pStcShutterShow->SetWindowText(strTemp);

	// Get current value of ExposureTime and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_EXPOSURE_TIME, &dShutterValue);
	GX_VERIFY(emStatus);

	m_dShutterValue = dShutterValue;

	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the Gain control
 
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::InitGainUI()
{
	CStatic     *pStcGain     = (CStatic *)GetDlgItem(IDC_STC_GAIN);
	CEdit       *pEditGainVal = (CEdit *)GetDlgItem(IDC_EDIT_GAIN);

	// Check whether the edit control is available
	if ((pStcGain == NULL) || (pEditGainVal == NULL))
	{
		return;
	}
    
	GX_STATUS emStatus = GX_STATUS_ERROR;
	CString   strRange = "";
	double    dGainVal = 0.0;

	// Get the range of Gain value
	emStatus = GXGetFloatRange(m_hDevice, GX_FLOAT_GAIN, &m_stGainFloatRange);
	GX_VERIFY(emStatus);

	// Show the range in the static control
    strRange.Format("Gain(%.4f~%.4f)%s", m_stGainFloatRange.dMin, m_stGainFloatRange.dMax, m_stGainFloatRange.szUnit);
	pStcGain->SetWindowText(strRange);
	
	// Get current value of Gain and update to the UI
	emStatus = GXGetFloat(m_hDevice, GX_FLOAT_GAIN, &dGainVal);
	GX_VERIFY(emStatus);

	m_dGainValue = dGainVal;
   
	UpdateData(FALSE);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize the UI of ComboBox 
\param   emFeatureID   [in]    
\param   pComboBox     [in]   
\param   pIsImplement  [in]    
\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::InitEnumUI(GX_FEATURE_ID_CMD emFeatureID, CComboBox *pComboBox, bool bIsImplement)
{
	// Check whether the ComboBox is available
	if ((pComboBox == NULL) || !bIsImplement)
	{
		return;
	}

	GX_ENUM_DESCRIPTION *pEnum       = NULL;
	GX_STATUS emStatus    = GX_STATUS_ERROR;
	size_t    nbufferSize = 0;
	uint32_t  nEntryNum   = 0;
	int64_t   nEnumValue  = -1;
	int       nCurcel     = 0;

	// Get the enumeration entry the device supports.
	emStatus = GXGetEnumEntryNums(m_hDevice, emFeatureID, &nEntryNum);
    GX_VERIFY(emStatus);

	// Allocate memory for getting the description of the enumeration entry
	nbufferSize = nEntryNum * sizeof(GX_ENUM_DESCRIPTION);
	pEnum       = new GX_ENUM_DESCRIPTION[nEntryNum];
	if (pEnum == NULL)
	{
		MessageBox("Fail to allocate memory!");
		return;
	}

	emStatus = GXGetEnumDescription(m_hDevice, emFeatureID, pEnum, &nbufferSize);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}

		return;
	}

	// Get the enumeration value and show it to the UI 
	emStatus = GXGetEnum(m_hDevice, emFeatureID, &nEnumValue);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		if (pEnum != NULL)
		{  
			delete []pEnum;
			pEnum = NULL;
		}

		return;
	}

    // Initialize the options of the ComboBox
	pComboBox->ResetContent();
	for (uint32_t i = 0; i < nEntryNum; i++)
	{	
		pComboBox->SetItemData(pComboBox->AddString(pEnum[i].szSymbolic), (uint32_t)pEnum[i].nValue);
		if (pEnum[i].nValue == nEnumValue)
		{
			nCurcel = i;
		}
	}

	// Show the value to the UI
	pComboBox->SetCurSel(nCurcel);

	// Release memory 
	if (pEnum != NULL)
	{  
		delete []pEnum;
		pEnum = NULL;
    }
}

//---------------------------------------------------------------------------------
/**
\brief   This is a acquisition callback function that demonstrates how to process the acquired image
\param   pFrame                          

\return  void
*/
//----------------------------------------------------------------------------------
void __stdcall CGxSingleCamMonoDlg::OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame)
{
	CGxSingleCamMonoDlg *pDlg = (CGxSingleCamMonoDlg*)(pFrame->pUserParam);
	int nImageHeight = (int)pDlg->m_nImageHeight;
	int nImageWidth  = (int)pDlg->m_nImageWidth;

	if (pFrame->status == 0)
	{
		memcpy(pDlg->m_pBufferRaw,pFrame->pImgBuf,pFrame->nImgSize);

		// Show image after flipping image data if mono camera has been used.
		for(int i =0;i <nImageHeight;i++)
		{
			memcpy(pDlg->m_pImageBuffer+i*nImageWidth, pDlg->m_pBufferRaw+(nImageHeight-i-1)*nImageWidth,(size_t)nImageWidth);
		}

	    pDlg->DrawImg();

		if (pDlg->m_bIsSaveImg)
		{
			pDlg->SaveImage();
		}
	}
}

//---------------------------------------------------------------------------------
/**
\brief   Draw image in the window

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::DrawImg()
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
		           (int)m_nImageWidth,
		           (int)m_nImageHeight,
		            m_pImageBuffer,
		            m_pBmpInfo,
		            DIB_RGB_COLORS,
		            SRCCOPY
		            );
}

//---------------------------------------------------------------------------------
/**
\brief   Release resources for showing image 

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::UnPrepareForShowImg()
{
	if (m_pBufferRaw != NULL)
	{
		delete []m_pBufferRaw;
    	m_pBufferRaw = NULL;
	}
	
	if (m_pImageBuffer != NULL)
	{
		delete []m_pImageBuffer;
		m_pImageBuffer = NULL;
	}
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to send TriggerSoftware command.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnBnClickedBtnSofttrigger() 
{
	GX_STATUS emStatus = GX_STATUS_ERROR;
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_TRIGGER_SOFTWARE);

	GX_VERIFY(emStatus);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the TriggerMode

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnCbnSelchangeComboTriggerMode() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	bool bIsWritable = TRUE;
	int64_t  nCurrentEnumValue = 0;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_MODE);

	int      nIndex    = pBox->GetCurSel();                     // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);    // Get the enumeration entry of current selection

	emStatus = GXIsWritable(m_hDevice,GX_ENUM_TRIGGER_MODE,&bIsWritable);
	GX_VERIFY(emStatus);

	emStatus = GXGetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, &nCurrentEnumValue);
	GX_VERIFY(emStatus);

	if (bIsWritable)
	{
		emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_MODE, nEnumVal);
		GX_VERIFY(emStatus);

		// Change the TriggerMode 
		m_nTriggerMode = nEnumVal;
	}
	else
	{
		MessageBox("The current state is not writable,please stop acquisition and reset the device!");
		if (nCurrentEnumValue != nEnumVal)
		{
			if (GX_TRIGGER_MODE_ON == nEnumVal)
			{
				for (int i = 0;i < pBox->GetCount();i++)
				{
					if (GX_TRIGGER_MODE_OFF == pBox->GetItemData(i))
					{
						pBox->SetCurSel(i);
						break;
					}
				}
			}
			else
			{
				for (int i = 0;i < pBox->GetCount();i++)
				{
					if (GX_TRIGGER_MODE_ON == pBox->GetItemData(i))
					{
						pBox->SetCurSel(i);
						break;
					}
				}
			}
		}
		else
		{
			return;
		}
	}

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to save the BMP images.

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::SaveImage()
{
	int64_t		  nImageSize  = m_nImageHeight * m_nImageWidth;  
	CString       strFileName = "";
	BITMAPINFO*	  pbitmapinfo = m_pBmpInfo;
	SYSTEMTIME    sysTime;

	
   	BOOL bRet = CreateDirectory(m_strFilePath, NULL);
	if (bRet)
	{
		SetFileAttributes(m_strFilePath, FILE_ATTRIBUTE_NORMAL);
	}

	// Get current time as the filename of images
	 GetLocalTime(&sysTime);
	strFileName.Format("%s\\%d_%d_%d_%d_%d_%d_%d.bmp",m_strFilePath,
		                                              sysTime.wYear,
												      sysTime.wMonth,
		                                              sysTime.wDay,
												      sysTime.wHour,
												      sysTime.wMinute,
												      sysTime.wSecond,
												      sysTime.wMilliseconds);

	BITMAPFILEHEADER	stBfh	    = {0};
	DWORD				dwBytesRead = 0;

	stBfh.bfType	= (WORD)'M' << 8 | 'B';			
	stBfh.bfOffBits	= sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (256 * 4); 
	stBfh.bfSize    = stBfh.bfOffBits + (DWORD)nImageSize;		

	
	HANDLE hFile = ::CreateFile(strFileName,
		                        GENERIC_WRITE,   
		                        0,
		                        NULL,
		                        CREATE_ALWAYS,													
		                        FILE_ATTRIBUTE_NORMAL,
		                        NULL
		                        );
	if (hFile != INVALID_HANDLE_VALUE) 
	{
		::WriteFile(hFile, &stBfh, sizeof(BITMAPFILEHEADER), &dwBytesRead, NULL);
		::WriteFile(hFile, m_pBmpInfo, sizeof(BITMAPINFOHEADER) + (256 * 4), &dwBytesRead, NULL); 
		::WriteFile(hFile, m_pImageBuffer, (DWORD)nImageSize, &dwBytesRead, NULL);

		CloseHandle(hFile);
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in]  
 
\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::ShowErrorString(GX_STATUS emErrorStatus)
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

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to use the control to update UI 

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::UpDateUI()
{
	GetDlgItem(IDC_BTN_OPEN_DEVICE)->EnableWindow(!m_bDevOpened);
	GetDlgItem(IDC_BTN_CLOSE_DEVICE)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_BTN_START_SNAP)->EnableWindow(m_bDevOpened && !m_bIsSnap);
	GetDlgItem(IDC_BTN_STOP_SNAP)->EnableWindow(m_bDevOpened && m_bIsSnap);
	GetDlgItem(IDC_BTN_SOFTTRIGGER)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_EDIT_SHUTTER)->EnableWindow(m_bDevOpened);
	GetDlgItem(IDC_EDIT_GAIN)->EnableWindow(m_bDevOpened);

	GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE)->EnableWindow(m_bDevOpened && m_bTriggerActive);
	GetDlgItem(IDC_COMBO_TRIGGER_MODE)->EnableWindow(m_bDevOpened &&  m_bTriggerMode);
	GetDlgItem(IDC_COMBO_TRIGGER_SOURCE)->EnableWindow(m_bDevOpened && m_bTriggerSource);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnBnClickedBtnOpenDevice()
{
	// TODO: Add your control notification handler code here
	GX_STATUS     emStatus = GX_STATUS_SUCCESS;
	uint32_t      nDevNum  = 0;
	GX_OPEN_PARAM stOpenParam;

	
	stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
	stOpenParam.openMode   = GX_OPEN_INDEX;
	stOpenParam.pszContent = "1";
              
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
	emStatus = GXOpenDevice(&stOpenParam, &m_hDevice);
	GX_VERIFY(emStatus);
    m_bDevOpened = TRUE;

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

	InitUI();

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to stop acquiring image and close device.
 
 \return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnBnClickedBtnCloseDevice()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// If the device is acquiring image then stop it.
	if (m_bIsSnap)
	{
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

		m_bIsSnap = FALSE;

	    // Release memory for showing image
		UnPrepareForShowImg();
	}

	emStatus = GXCloseDevice(m_hDevice);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here 
	}

	m_bDevOpened = FALSE;
	m_hDevice    = NULL;

	UpDateUI();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to stop acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnBnClickedBtnStopSnap()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Send AcquisitionStop command 
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_STOP);
	GX_VERIFY(emStatus);

	//Unregister the CaptureCallback function
	emStatus = GXUnregisterCaptureCallback(m_hDevice);
	GX_VERIFY(emStatus);
   
	m_bIsSnap = FALSE;

	// Release memory for showing image
	UnPrepareForShowImg();

	UpDateUI();
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to initialize device

\return  void
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxSingleCamMonoDlg::InitDevice()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	//Set the continuous frame acquisition mode
	emStatus = GXSetEnum(m_hDevice,GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
	VERIFY_STATUS_RET(emStatus);

	// If the device supports the pixel format of"GX_PIXEL_FORMAT_BAYER_GR8",
	// the following code can be used to set the pixel format to GX_PIXEL_FORMAT_BAYER_GR8:
	// emStatus = GXSetEnum(m_hDevice, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_GR8);
	// VERIFY_STATUS_RET(emStatus);

	// If you don't know the device supports which kind of pixel format ,you could use following function to set the pixel format to 8-bit.
	emStatus = SetPixelFormat8bit(); 
	VERIFY_STATUS_RET(emStatus);

	return emStatus;
}

// ---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the pixel format to 8bit

\return  emStatus GX_STATUS_SUCCESS: Success, else: Fail
*/
// ----------------------------------------------------------------------------------
GX_STATUS CGxSingleCamMonoDlg::SetPixelFormat8bit()
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
			if (pEnumDescription == NULL)
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
\brief  This function demonstrates how to get the parameters of the device

\return void
*/
//----------------------------------------------------------------------------------
GX_STATUS CGxSingleCamMonoDlg::GetDeviceParam()
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
	return emStatus;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for showing color image.

\return true:Success false:Fail
*/
//----------------------------------------------------------------------------------
bool  CGxSingleCamMonoDlg::PrepareForShowImg()
{	
	//---------------------------------------------------------------------
	//----------------------Initialize bitmap header-----------------------
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
		m_pBmpInfo->bmiColors[i].rgbBlue	=i;
		m_pBmpInfo->bmiColors[i].rgbGreen	=i;
		m_pBmpInfo->bmiColors[i].rgbRed		=i;
		m_pBmpInfo->bmiColors[i].rgbReserved=i;
	}

	//--------------------------------------------------------------------------
	//-------------------Allocate buffer for saving image data------------------
	//Allocate memory for showing mono images.
	m_pBufferRaw = new BYTE[(size_t)m_nPayLoadSize];
	if (m_pBufferRaw == NULL)
	{
		return false;
	}

	//Allocate memory for showing image after flipping
	m_pImageBuffer = new BYTE[(size_t)(m_nImageWidth * m_nImageHeight)];
	if (m_pImageBuffer == NULL)
	{
		delete []m_pBufferRaw;
		m_pBufferRaw = NULL;

		return false;
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to set the switch for saving the BMP images

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnBnClickedCheckSaveImage()
{
	// TODO: Add your control notification handler code here
	m_bIsSaveImg = !m_bIsSaveImg;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to start acquisition

\return void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnBnClickedBtnStartSnap()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_ERROR;

	if (!PrepareForShowImg())
	{
		MessageBox("Fail to allocate resource for showing image!");
        return;
	}
	
	//Register the CaptureCallback function 
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

	//Send AquisitionStart command
	emStatus = GXSendCommand(m_hDevice, GX_COMMAND_ACQUISITION_START);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		UnPrepareForShowImg();
		ShowErrorString(emStatus);
		return;
	}

	m_bIsSnap = TRUE;

	UpDateUI();
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the TriggerActivation

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnCbnSelchangeComboTriggerActive()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_ACTIVE);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection
    
	//Enumerate and change the TriggerActivation
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_ACTIVATION, nEnumVal);
    GX_VERIFY(emStatus);
}

//---------------------------------------------------------------------------------
/**
\brief   This function demonstrates how to change the TriggerResource

\return  void
*/
//----------------------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnCbnSelchangeComboTriggerSource()
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	CComboBox *pBox    = (CComboBox *)GetDlgItem(IDC_COMBO_TRIGGER_SOURCE);

	int      nIndex    = pBox->GetCurSel();                   // Get current selection
	int64_t  nEnumVal  = (int64_t)pBox->GetItemData(nIndex);  // Get the enumeration entry of current selection

	//Enumerate and change the TriggerResource
	emStatus = GXSetEnum(m_hDevice, GX_ENUM_TRIGGER_SOURCE, nEnumVal);
	GX_VERIFY(emStatus);
}

//-----------------------------------------------------------------------
/**
\brief    This function demonstrates how to set the ExposureTime

\return   void
*/
//-----------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnKillfocusEditShutter() 
{
	// TODO: Add your control notification handler code here
	//Check whether the handle is available
	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData();
	GX_STATUS status = GX_STATUS_SUCCESS;

	if (m_dShutterValue > m_stShutterFloatRange.dMax)
	{
		m_dShutterValue = m_stShutterFloatRange.dMax;
	}

	if (m_dShutterValue < m_stShutterFloatRange.dMin)
	{
		m_dShutterValue = m_stShutterFloatRange.dMin;
	}

	//Set the ExposureTime
	status = GXSetFloat(m_hDevice,GX_FLOAT_EXPOSURE_TIME,m_dShutterValue);
	GX_VERIFY(status);

	UpdateData(FALSE);
}

//-----------------------------------------------------------------------
/**
\brief   This function demonstrates how to set the Gain

\return  void
*/
//-----------------------------------------------------------------------
void CGxSingleCamMonoDlg::OnKillfocusEditGain() 
{
	// TODO: Add your control notification handler code here
	//Check whether the handle is available
	if (m_hDevice == NULL)
	{
		return;
	}

	UpdateData();
	GX_STATUS status = GX_STATUS_SUCCESS;

	if (m_dGainValue > m_stGainFloatRange.dMax)
	{
		m_dGainValue = m_stGainFloatRange.dMax;
	}
	
	if (m_dGainValue < m_stGainFloatRange.dMin)
	{
		m_dGainValue = m_stGainFloatRange.dMin;
	}
	
	status = GXSetFloat(m_hDevice, GX_FLOAT_GAIN, m_dGainValue);
	GX_VERIFY(status);

	UpdateData(FALSE);
}

//-----------------------------------------------------------------------
/**
\brief   This function demonstrates how to deal with message of "Enter" 
\param   pMsg 

\return  Success: TRUE Fail:FALSE
*/
//-----------------------------------------------------------------------
BOOL CGxSingleCamMonoDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	CWnd *pWnd = NULL;
	int   nID = 0;         
	GX_STATUS status = GX_STATUS_SUCCESS;

	if (WM_KEYDOWN == pMsg->message && VK_RETURN == pMsg->wParam)
	{
		pWnd = GetFocus();

		nID = pWnd->GetDlgCtrlID();

		switch (nID)
		{

		case IDC_EDIT_SHUTTER:

		case IDC_EDIT_GAIN:
			
			SetFocus();

			break;

		default:
			break;
		}

		return TRUE;
	}

	if ((WM_KEYDOWN == pMsg->message) && (VK_ESCAPE == pMsg->wParam))
	{
		return TRUE;
	}

	return CDialog::PreTranslateMessage(pMsg);
}
