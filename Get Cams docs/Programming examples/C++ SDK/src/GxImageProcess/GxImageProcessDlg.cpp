// GxImageProcessDlg.cpp : implementation file
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include "GxImageProcess.h"
#include "GxImageProcessDlg.h"
#include "FileVersion.h"

using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

class CAboutDlg : public CDialog
{
public:
	CAboutDlg();

// Dialog Data
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
public:
	CString m_strProtect_Version;
	CString m_strLegalCopyRight;
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
, m_strProtect_Version(_T(""))
, m_strLegalCopyRight(_T(""))
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_STATIC_PRODUCTVERSION, m_strProtect_Version);
	DDX_Text(pDX, IDC_STATIC_LEGALCOPYRIGHT, m_strLegalCopyRight);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()

// CGxImageProcessDlg dialog
CGxImageProcessDlg::CGxImageProcessDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxImageProcessDlg::IDD, pParent)
	, m_bCheckAcceleract(FALSE)
	, m_bCheckPixelCorrect(FALSE)
	, m_bCheckDenoise(FALSE)
	, m_bCheckColorCorrect(FALSE)
	, m_bCheckSharpen(FALSE)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);              

	m_bIsOpen                    = false;
	m_bIsSnap                    = false;
	m_bIsWhiteBalance            = false;
	m_bIsWhiteBalanceLight       = false;
    m_bEnableColorCorrect        = false;
    m_bEnableGamma               = false;
    m_bEnableSharpness           = false;
	m_bEnableStaturation         = false; 
	m_bEnableCheckDenoise        = false; 
	m_bIsColorFilter             = false;              
	m_bEnumDevices               = false;
	m_pWnd                       = NULL;
	m_pBitmap                    = NULL;
	m_pSampleCaptureEventHandler = NULL;
	m_nComBoAWBOldValue          = 0;
	m_nComBoAWBLightOldValue     = 0;
	m_nOldSliderContrastValue    = 0;
	m_nOldSliderGammaValue       = 0;
	m_nOldSliderLightnessValue   = 0;
	m_nOldSliderSaturationValue  = 0;
	m_nOldSliderSharpenValue     = 0;
	m_strBalanceWhiteAutoValue   = "Off";
	m_vecDeviceInfo.clear();
	
}

void CGxImageProcessDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SLIDER_SHARPEN, m_sliderSharpen);
	DDX_Control(pDX, IDC_SLIDER_GAMMA, m_sliderGamma);
	DDX_Control(pDX, IDC_SLIDER_CONTRAST, m_sliderContrast);
	DDX_Control(pDX, IDC_SLIDER_LIGHTNESS, m_sliderLightness);
	DDX_Control(pDX, IDC_SLIDER_SATURATION, m_sliderSaturation);
	DDX_Check(pDX, IDC_AIDC_OPEN_ACCELERATECCELERATE_CHECK, m_bCheckAcceleract);
	DDX_Check(pDX, IDC_OPEN_BAD_PIXEL_CORRECT, m_bCheckPixelCorrect);
	DDX_Check(pDX, IDC_OPEN_DENOISE, m_bCheckDenoise);
	DDX_Check(pDX, IDC_OPEN_COLOR_CORRECT, m_bCheckColorCorrect);
	DDX_Check(pDX, IDC_OPEN_SHARPEN, m_bCheckSharpen);
	DDX_Control(pDX, IDC_COMBO_ChooseDevice, m_comboChooseDevice);
}

BEGIN_MESSAGE_MAP(CGxImageProcessDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP

ON_WM_CLOSE()
//ON_WM_HSCROLL()
//ON_NOTIFY(NM_CUSTOMDRAW, IDC_SHARPEN_SLIDER, &CGxImageProcessDlg::OnNMCustomdrawSharpenSlider)
ON_WM_HSCROLL()
ON_BN_CLICKED(IDC_OPEN_DEVICE, &CGxImageProcessDlg::OnBnClickedOpenDevice)
ON_BN_CLICKED(IDC_CLOSE_DEVICE, &CGxImageProcessDlg::OnBnClickedCloseDevice)
ON_BN_CLICKED(IDC_START_SNAP, &CGxImageProcessDlg::OnBnClickedStartSnap)
ON_BN_CLICKED(IDC_STOP_SNAP, &CGxImageProcessDlg::OnBnClickedStopSnap)
ON_CBN_SELCHANGE(IDC_COMBO_AWB, &CGxImageProcessDlg::OnCbnSelchangeComboAwb)
ON_CBN_SELCHANGE(IDC_COMBO_AWB_LIGHT, &CGxImageProcessDlg::OnCbnSelchangeComboAwbLight)
ON_BN_CLICKED(IDC_AIDC_OPEN_ACCELERATECCELERATE_CHECK, &CGxImageProcessDlg::OnBnClickedAidcOpenAccelerateccelerateCheck)
ON_BN_CLICKED(IDC_OPEN_BAD_PIXEL_CORRECT, &CGxImageProcessDlg::OnBnClickedOpenBadPixelCorrect)
ON_BN_CLICKED(IDC_OPEN_DENOISE, &CGxImageProcessDlg::OnBnClickedOpenDenoise)
ON_BN_CLICKED(IDC_OPEN_COLOR_CORRECT, &CGxImageProcessDlg::OnBnClickedOpenColorCorrect)
ON_BN_CLICKED(IDC_OPEN_SHARPEN, &CGxImageProcessDlg::OnBnClickedOpenSharpen)
ON_CBN_DROPDOWN(IDC_COMBO_AWB, &CGxImageProcessDlg::OnCbnDropdownComboAwb)
ON_CBN_DROPDOWN(IDC_COMBO_AWB_LIGHT, &CGxImageProcessDlg::OnCbnDropdownComboAwbLight)
ON_WM_TIMER()
END_MESSAGE_MAP()


// CGxImageProcessDlg message handlers

BOOL CGxImageProcessDlg::OnInitDialog()
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

	try
	{
		// Initialize GxIAPICPP
		IGXFactory::GetInstance().Init();

		// Get the pointer to the image display window
		m_pWnd = GetDlgItem(IDC_SHOW_PICTURE_STATIC);

		// Allocate memory for the callback pointer
		m_pSampleCaptureEventHandler = new CSampleCaptureEventHandler();

		// Set Timer
		SetTimer(0,1000,NULL);

		//Initializes the drop-down box
		__InitCombo();

		__UpdateUI();

	}
	catch (CGalaxyException& e)
	{
		if (m_pSampleCaptureEventHandler != NULL)
		{
			delete m_pSampleCaptureEventHandler;
			m_pSampleCaptureEventHandler = NULL;
		}
		MessageBox(e.what());
		return FALSE;
	}
	catch (std::exception& e)
	{
		if (m_pSampleCaptureEventHandler != NULL)
		{
			delete m_pSampleCaptureEventHandler;
			m_pSampleCaptureEventHandler = NULL;
		}
		MessageBox(e.what());
		return FALSE;
	}

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxImageProcessDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		char chFileName[MAX_PATH] = {'\0'};
		GetModuleFileName(NULL,(LPTSTR)chFileName, MAX_PATH);
		CString strFileName = "";
		strFileName.Format("%s",chFileName);
		CFileVersion fv(chFileName);
		CAboutDlg dlgAbout;
		dlgAbout.m_strProtect_Version = _T("Version: ") + fv.GetProductVersion();
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

void CGxImageProcessDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

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

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CGxImageProcessDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

//----------------------------------------------------------------
/**
\brief  The initial drop-down list control
*/
//----------------------------------------------------------------
void CGxImageProcessDlg::__InitCombo()
{

	IGXFactory::GetInstance().UpdateDeviceList(1000, m_vecDeviceInfo);

	// Determine the number of devices currently connected
	if (m_vecDeviceInfo.size() <= 0)
	{
		MessageBox("Not enumerated to device, please insert camera and restart the program!");
		m_bEnumDevices = false;
		__UpdateUI();
		return;
	}
	else
	{
		m_bEnumDevices = true;
		__UpdateUI();

		for (uint32_t i = 0; i < m_vecDeviceInfo.size(); i++)
		{
			m_comboChooseDevice.AddString(m_vecDeviceInfo[i].GetDisplayName());
		}

		// The default drop-down list selects the first camera
		m_comboChooseDevice.SetCurSel(0);
	}
}
void CGxImageProcessDlg::OnBnClickedOpenDevice()
{
	// TODO: Add your control notification handler code here
	bool bIsDeviceOpen = false;          ///< The flag indicates whether the device has been opened
	bool bIsStreamOpen = false;          ///< The flag indicates whether the stream has been opened

	try
	{
		int nSel = m_comboChooseDevice.GetCurSel();
		if (nSel < 0)
		{
			MessageBox("Please insert camera!");
			return;
		}

		// Open Device
		m_objDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(m_vecDeviceInfo[nSel].GetSN(), GX_ACCESS_EXCLUSIVE);
		bIsDeviceOpen =true;
		m_objFeatureControlPtr = m_objDevicePtr->GetRemoteFeatureControl();

		// Create the image processing configuration parameter object
		m_objImageProcessPtr = m_objDevicePtr->CreateImageProcessConfig();

		//Check whether support Bayer format
		m_bIsColorFilter = m_objFeatureControlPtr->IsImplemented("PixelColorFilter");

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}
		m_pBitmap = new CGXBitmap(m_objDevicePtr, m_pWnd);

		// Get the number of stream channels
		int32_t nStreamCount = m_objDevicePtr->GetStreamCount();
		if (nStreamCount > 0)
		{
			m_objStreamPtr = m_objDevicePtr->OpenStream(0);
			m_objStreamFeatureControlPtr = m_objStreamPtr->GetFeatureControl();
			bIsStreamOpen  = true;
		}
		else
		{
			throw exception("No streams found!");
		}

        //It is recommended that the user set the camera's stream channel packet length value 
        //according to the current network environment after turning on 
        //the network camera to improve the collection performance of the network camera. 
        //For the setting method, refer to the following code.
        GX_DEVICE_CLASS_LIST objDeviceClass = m_objDevicePtr->GetDeviceInfo().GetDeviceClass();
        if(GX_DEVICE_CLASS_GEV == objDeviceClass)
        {
            // Determine whether the device supports the stream channel packet function.
            if(true == m_objFeatureControlPtr->IsImplemented("GevSCPSPacketSize"))
            {
                // Get the optimal packet length value of the current network environment
                int nPacketSize = m_objStreamPtr->GetOptimalPacketSize();
                // Set the optimal packet length value to the stream channel packet length of the current device.
                m_objFeatureControlPtr->GetIntFeature("GevSCPSPacketSize")->SetValue(nPacketSize);
            } 
        }

		// initialize device param
		__InitParam();

        // initialize UI
		__InitUI();
		m_bIsOpen = true;
		m_bEnumDevices = false;

		// Update UI
		__UpdateUI();
	}
	catch (CGalaxyException& e)
	{
		// Check whether the stream has been opened
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		// Check whether the device has been opened
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}

		MessageBox(e.what());
		return;
	}
	catch (std::exception& e)
	{
		// Check whether the stream has been opened
		if (bIsStreamOpen)
		{
			m_objStreamPtr->Close();
		}

		// Check whether the device has been opened
		if (bIsDeviceOpen)
		{
			m_objDevicePtr->Close();
		}

		if (m_pBitmap != NULL)
		{
			delete m_pBitmap;
			m_pBitmap = NULL;
		}

		MessageBox(e.what());
		return;
	}
}

//---------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize device 

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::__InitParam()
{
	bool bBalanceWhiteAutoRead = false;         ///< The flag indicates whether BalanceWhite is read

	// Set the TriggerMode off
	m_objFeatureControlPtr->GetEnumFeature("TriggerMode")->SetValue("Off");

	// Set the continuous frame acquisition mode
	m_objFeatureControlPtr->GetEnumFeature("AcquisitionMode")->SetValue("Continuous");

	// Check whether the device supports BalanceWhiteAuto 
	m_bIsWhiteBalance = m_objFeatureControlPtr->IsImplemented("BalanceWhiteAuto");

	// Check whether BalanceWhite can be read
	bBalanceWhiteAutoRead   = m_objFeatureControlPtr->IsReadable("BalanceWhiteAuto");

	if (m_bIsWhiteBalance && bBalanceWhiteAutoRead)
	{
			//Get Current BalanceWhite
			m_strBalanceWhiteAutoValue = m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")
				                                               ->GetValue();
	}


	// Check whether the device supports WhiteBalanceLight
	m_bIsWhiteBalanceLight = m_objFeatureControlPtr->IsImplemented("AWBLampHouse");

    __IsEnableDeviceColorCorrection();
    __IsEnableDeviceGamma();
    __IsEnableDeviceSharpen();
}

//---------------------------------------------------------------------
/**
\brief	This function demonstrates how to initialize device 

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::__InitUI()
{
	char chPutText[64] = {0};

	// The following constants define reference <<C++ SDK Programming Reference Manual>> ,Chapter 2
	const double dSharpenParamMax    = 5;     ///< Maximum sharpen
	const double dSharpenParamMin    = 0.1;   ///< Minimum sharpen
	const double dGammmaParamMax     = 10;    ///< Maximum Gamma
	const double dGammaParamMin      = 0.1;   ///< Minimum Gamma
	const int nContrastParamMax      = 100;   ///< Maximum Contrast
	const int nContrastParamMin      = -50;   ///< Minimum Contrast
	const int nLightnessParamMax     = 150;   ///< Maximum Lightness
	const int nLightnessParamMin     = -150;  ///< Minimum Lightness
	const int nSaturationParamMax    = 128;   ///< Maximum Saturation
	const int nSaturationParamMin    = 0;     ///< Minimum Saturation

	// Initialize BalanceWhite control
	m_nComBoAWBOldValue = __InitEnumUI("BalanceWhiteAuto", 
		                               (CComboBox *)GetDlgItem(IDC_COMBO_AWB),
									    m_bIsWhiteBalance);

	// Initialize BalanceWhite light control
    m_nComBoAWBLightOldValue = __InitEnumUI("AWBLampHouse",
		                                     (CComboBox *)GetDlgItem(IDC_COMBO_AWB_LIGHT),
											  m_bIsWhiteBalanceLight);

	m_sliderSharpen.SetRange(int(dSharpenParamMin * PARAM_TRANSFORMATION_VALUE),
		                      int(dSharpenParamMax * PARAM_TRANSFORMATION_VALUE),TRUE);

	m_sliderGamma.SetRange(int(dGammaParamMin * PARAM_TRANSFORMATION_VALUE),
		                    int(dGammmaParamMax * PARAM_TRANSFORMATION_VALUE),TRUE);

	m_sliderContrast.SetRange(nContrastParamMin,nContrastParamMax,TRUE);
	m_sliderLightness.SetRange(nLightnessParamMin,nLightnessParamMax,TRUE);
	m_sliderSaturation.SetRange(nSaturationParamMin,nSaturationParamMax,TRUE);
	m_sliderSharpen.SetPos(int(m_objImageProcessPtr->GetSharpenParam()*PARAM_TRANSFORMATION_VALUE));
	m_sliderGamma.SetPos(int(m_objImageProcessPtr->GetGammaParam()*PARAM_TRANSFORMATION_VALUE));
	m_sliderContrast.SetPos(m_objImageProcessPtr->GetContrastParam());
	m_sliderLightness.SetPos(m_objImageProcessPtr->GetLightnessParam());
	

	// Initialize Edit control
	sprintf_s(chPutText,sizeof(chPutText),"%f",m_objImageProcessPtr->GetSharpenParam());
	GetDlgItem(IDC_SHARPEN_EDIT)->SetWindowText(chPutText);
	sprintf_s(chPutText,sizeof(chPutText),"%f",m_objImageProcessPtr->GetGammaParam());
	GetDlgItem(IDC_GAMMA_EDIT)->SetWindowText(chPutText);
	sprintf_s(chPutText,sizeof(chPutText),"%d",m_objImageProcessPtr->GetContrastParam());
	GetDlgItem(IDC_CONTRASE_EDIT)->SetWindowText(chPutText);
	sprintf_s(chPutText,sizeof(chPutText),"%d",m_objImageProcessPtr->GetLightnessParam());
	GetDlgItem(IDC_BRIGHT_EDIT)->SetWindowText(chPutText);
	

	// Initialize Check control
	m_bCheckAcceleract   = m_objImageProcessPtr->IsAccelerate();
	m_bCheckPixelCorrect = m_objImageProcessPtr->IsDefectivePixelCorrect();
	m_bCheckSharpen      = m_objImageProcessPtr->IsSharpen();


	if(m_bIsColorFilter)
	{
		m_sliderSaturation.SetPos(m_objImageProcessPtr->GetSaturationParam());
		sprintf_s(chPutText,sizeof(chPutText),"%d",m_objImageProcessPtr->GetSaturationParam());
		m_bCheckDenoise      = m_objImageProcessPtr->IsDenoise();
	    m_bCheckColorCorrect = m_objImageProcessPtr->IsColorCorrection();
		m_bEnableStaturation = false;
		m_bEnableCheckDenoise = false;
		m_bEnableColorCorrect = false;

	}else
	{
		m_sliderSaturation.SetPos(nSaturationParamMin);
		sprintf_s(chPutText,sizeof(chPutText),"%d",nSaturationParamMin);
		m_bEnableStaturation = true;
		m_bEnableCheckDenoise = true;
		m_bEnableColorCorrect = true;
	}

	GetDlgItem(IDC_SATURA_EDIT)->SetWindowText(chPutText);

	UpdateData(FALSE);

}

//-----------------------------------------------------------------------
/**
\brief   This function demonstrates how to ComboBox
\param   strFeaturName    Function name
\param   pComboBox        Control pointer
\param   bIsImplemented   Whether support
\return  -1: Error information  other value: enum value
*/
//-----------------------------------------------------------------------
int CGxImageProcessDlg::__InitEnumUI(const gxstring& strFeaturName, CComboBox *pComboBox, bool bIsImplemented)
{
	// Check whether the ComboBox is available
	if (pComboBox == NULL || (!bIsImplemented))
	{
		return -1;
	}
	GxIAPICPP::gxstring_vector vectorEnumEntryList;    ///< Enumeration container
	string strCurrentEnumEntryList = "";               ///< Current enum value
	int    nCursel = 0;
	CEnumFeaturePointer objEnumFeaturePtr = m_objFeatureControlPtr->GetEnumFeature(strFeaturName);

	// Clear control
	pComboBox->ResetContent();

	// Get current value
	strCurrentEnumEntryList = objEnumFeaturePtr->GetValue().c_str();

	// Get the enumeration entry the device supports.
	vectorEnumEntryList = objEnumFeaturePtr->GetEnumEntryList();
	for (uint32_t i = 0; i<vectorEnumEntryList.size(); i++)
	{
		pComboBox->SetItemData(pComboBox->AddString(vectorEnumEntryList[i].c_str()), i);
		if (strCurrentEnumEntryList == vectorEnumEntryList[i].c_str())
		{
			nCursel = i;
		}
	}
	pComboBox->SetCurSel(nCursel);
	return nCursel;
}

//---------------------------------------------------------------------
/**
\brief  Update UI

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::__UpdateUI()
{
	GetDlgItem(IDC_OPEN_DEVICE)->EnableWindow(m_bEnumDevices);
	GetDlgItem(IDC_CLOSE_DEVICE)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_START_SNAP)->EnableWindow(m_bIsOpen && !m_bIsSnap);
	GetDlgItem(IDC_STOP_SNAP)->EnableWindow(m_bIsOpen && m_bIsSnap);
	GetDlgItem(IDC_COMBO_AWB)->EnableWindow(m_bIsOpen && m_bIsWhiteBalance);
	GetDlgItem(IDC_COMBO_AWB_LIGHT)->EnableWindow(m_bIsOpen && m_bIsWhiteBalanceLight);
	GetDlgItem(IDC_AIDC_OPEN_ACCELERATECCELERATE_CHECK)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_OPEN_BAD_PIXEL_CORRECT)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_OPEN_DENOISE)->EnableWindow(m_bIsOpen && !m_bEnableCheckDenoise);
	GetDlgItem(IDC_OPEN_COLOR_CORRECT)->EnableWindow(m_bIsOpen && !m_bEnableColorCorrect);
	GetDlgItem(IDC_OPEN_SHARPEN)->EnableWindow(m_bIsOpen && !m_bEnableSharpness);
	GetDlgItem(IDC_SLIDER_SHARPEN)->EnableWindow(m_bIsOpen && !m_bEnableSharpness && m_bCheckSharpen);
	GetDlgItem(IDC_SLIDER_GAMMA)->EnableWindow(m_bIsOpen && !m_bEnableGamma);
	GetDlgItem(IDC_SLIDER_CONTRAST)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_SLIDER_LIGHTNESS)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_SLIDER_SATURATION)->EnableWindow(m_bIsOpen && !m_bEnableStaturation);
	GetDlgItem(IDC_EDIT_SATURATION)->EnableWindow(m_bIsOpen && !m_bEnableStaturation);
	GetDlgItem(IDC_EDIT_SHARPEN)->EnableWindow(m_bIsOpen && !m_bEnableSharpness && m_bCheckSharpen);
	GetDlgItem(IDC_EDIT_GAMMA)->EnableWindow(m_bIsOpen && !m_bEnableGamma);
	GetDlgItem(IDC_EDIT_LIGHTNESS)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_EDIT_CONTRAST)->EnableWindow(m_bIsOpen);
	GetDlgItem(IDC_COMBO_ChooseDevice)->EnableWindow(m_bEnumDevices);

    ((CButton*)GetDlgItem(IDC_OPEN_COLOR_CORRECT))->SetCheck(m_bIsOpen && !m_bEnableColorCorrect);
}

void CGxImageProcessDlg::OnBnClickedCloseDevice()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Check whether the device has been stop acquisition
		if (m_bIsSnap)
		{
			// Send AcquisitionStop command
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

			// Stop Stream acquisition
			m_objStreamPtr->StopGrab();

			// Unregister the CaptureCallback function
			m_objStreamPtr->UnregisterCaptureCallback();
		}
	}
	catch(CGalaxyException)
	{
		//do nothing
	}

	try
	{
		// Close stream
		m_objStreamPtr->Close();
	}
	catch(CGalaxyException)
	{
		//do nothing
	}

	try
	{		
		// Close device
		m_objDevicePtr->Close();
		
	}
	catch (CGalaxyException)
	{
		//do nothing
	}

	m_bIsOpen = false;
	m_bIsSnap = false;
	m_bEnumDevices = true;

	if (m_pBitmap != NULL)
	{
		delete m_pBitmap;
		m_pBitmap = NULL;
	}

	// Update UI
	__UpdateUI();
}

void CGxImageProcessDlg::OnBnClickedStartSnap()
{
	// TODO: Add your control notification handler code here
    try
    {
		//Set StreamBufferHandlingMode
		m_objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("OldestFirst");

		// Register the CaptureCallback function
		m_objStreamPtr->RegisterCaptureCallback(m_pSampleCaptureEventHandler, this);

		// Start stream channel acquisition
		m_objStreamPtr->StartGrab(); 

		// Send AcquisitionStart command
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
		m_bIsSnap = true;

		// Update UI
		__UpdateUI();
		
    }
	catch (CGalaxyException& e)
	{
		MessageBox(e.what());
		return;
	}
	catch (std::exception& e)
	{
		MessageBox(e.what());
		return;
	}

}

void CGxImageProcessDlg::OnBnClickedStopSnap()
{
	// TODO: Add your control notification handler code here
	try
	{
		// Send AcquisitionStop command
		m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

		// Stop Stream acquisition
		m_objStreamPtr->StopGrab();

		// Unregister the CaptureCallback function
		m_objStreamPtr->UnregisterCaptureCallback();
		m_bIsSnap = false;

		// Update UI
		__UpdateUI();
		
	}
	catch (CGalaxyException& e)
	{
		MessageBox(e.what());
		return;
	}
	catch (std::exception& e)
	{
		MessageBox(e.what());
		return;
	}
}

void CGxImageProcessDlg::OnCbnSelchangeComboAwb()
{
	// TODO: Add your control notification handler code here
	CComboBox* pCombox = NULL;
	try
	{
		int         nIndex            = 0;
		CString     strCurrentContent = "";
		gxstring    strCurrentText    = "";
		pCombox = (CComboBox*)GetDlgItem(IDC_COMBO_AWB);
		nIndex = pCombox->GetCurSel();
		pCombox->GetLBText(nIndex,strCurrentContent);
		strCurrentText = strCurrentContent.GetBuffer(0);

		// Get the enumeration controller
		CEnumFeaturePointer objFeaturePtr = m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto");

		// Set BalanceWhite
		objFeaturePtr->SetValue(strCurrentText);
		m_nComBoAWBOldValue = nIndex;
		m_strBalanceWhiteAutoValue = strCurrentText;
	}
	catch (CGalaxyException& e)
	{
		pCombox->SetCurSel(m_nComBoAWBOldValue);
		MessageBox(e.what());
		return;
	}
	catch (std::exception& e)
	{
		pCombox->SetCurSel(m_nComBoAWBOldValue);
		MessageBox(e.what());
		return;
	}
}

void CGxImageProcessDlg::OnCbnSelchangeComboAwbLight()
{
	// TODO: Add your control notification handler code here
	CComboBox* pCombox = NULL;
	try
	{
		int         nIndex            = 0;
		CString     strCurrentContent = "";
		gxstring    strCurrentText    = "";
		pCombox = (CComboBox*)GetDlgItem(IDC_COMBO_AWB_LIGHT);
		nIndex = pCombox->GetCurSel();
		pCombox->GetLBText(nIndex,strCurrentContent);
		strCurrentText = strCurrentContent.GetBuffer(0);

		// Get the enumeration controller
		CEnumFeaturePointer objFeaturePtr = m_objFeatureControlPtr->GetEnumFeature("AWBLampHouse");

		// Set BalanceWhite
		objFeaturePtr->SetValue(strCurrentText);
		m_nComBoAWBLightOldValue = nIndex;
	}
	catch (CGalaxyException &e)
	{
		pCombox->SetCurSel(m_nComBoAWBLightOldValue);
		MessageBox(e.what());
		return;
	}
	catch (std::exception &e)
	{
		pCombox->SetCurSel(m_nComBoAWBLightOldValue);
		MessageBox(e.what());
		return;
	}
}

void CGxImageProcessDlg::OnBnClickedAidcOpenAccelerateccelerateCheck()
{
	// TODO: Add your control notification handler code here
	BOOL bAccelerateOld = m_bCheckAcceleract;      ///< Record BalanceWhite value 
	bool bFlag          = false;                   ///< The flag indicates whether the image height is an integer multiple of 4

	try
	{
		int64_t  nHeight = 0;        ///< Image height
		
        UpdateData(TRUE);

		//Get image height
		nHeight = m_objFeatureControlPtr->GetIntFeature("Height")->GetValue();
		bFlag = ((nHeight%4) != 0) && m_bCheckAcceleract;

		//Check whether the image height is an integer multiple of 4
		if (bFlag)
		{
			MessageBox("Use CPU acceleration,image height must be an integer multiple of 4!");
			m_bCheckAcceleract = FALSE;
			UpdateData(FALSE);
			return;
		}

		// Enable the algorithm to accelerate
		m_objImageProcessPtr->EnableAccelerate(m_bCheckAcceleract == TRUE);

	}
	catch (CGalaxyException& e)
	{

		m_bCheckAcceleract = bAccelerateOld;
		UpdateData(FALSE);
		MessageBox(e.what());
	}
	catch (std::exception& e)
	{
		m_bCheckAcceleract = bAccelerateOld;
		UpdateData(FALSE);
		MessageBox(e.what());
	}
   
	
}

void CGxImageProcessDlg::OnBnClickedOpenBadPixelCorrect()
{
	// TODO: Add your control notification handler code here
	BOOL bBadPixelCorrectOld = m_bCheckPixelCorrect;
	try
	{
		UpdateData(TRUE);

		// Enables bad pixel calibration
		m_objImageProcessPtr->EnableDefectivePixelCorrect(m_bCheckPixelCorrect == TRUE);
	}
	catch (CGalaxyException &e)
	{
		m_bCheckPixelCorrect = bBadPixelCorrectOld;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}
	catch (std::exception &e)
	{
		m_bCheckPixelCorrect = bBadPixelCorrectOld;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}


}

void CGxImageProcessDlg::OnBnClickedOpenDenoise()
{
	// TODO: Add your control notification handler code here
	BOOL bDenoise = m_bCheckDenoise;
	try
	{
		UpdateData(TRUE);

		// Enable denoise
		m_objImageProcessPtr->EnableDenoise(m_bCheckDenoise == TRUE);
	}
	catch (CGalaxyException &e)
	{
		m_bCheckDenoise = bDenoise;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}
	catch (std::exception &e)
	{
		m_bCheckDenoise = bDenoise;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}
}

void CGxImageProcessDlg::OnBnClickedOpenColorCorrect()
{
	// TODO: Add your control notification handler code here
	BOOL bColorCorrectOld = m_bCheckColorCorrect;
	try
	{
		UpdateData(TRUE);

		// Enable Color calibration
		m_objImageProcessPtr->EnableColorCorrection(m_bCheckColorCorrect == TRUE);
	}
	catch (CGalaxyException &e)
	{
		m_bCheckColorCorrect = bColorCorrectOld;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}
	catch (std::exception &e)
	{
		m_bCheckColorCorrect = bColorCorrectOld;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}
	
}

void CGxImageProcessDlg::OnBnClickedOpenSharpen()
{
	// TODO: Add your control notification handler code here
	BOOL bSharppenOld = m_bCheckSharpen;
	try
	{
		UpdateData(TRUE);

		// Enable sharpen
		m_objImageProcessPtr->EnableSharpen(m_bCheckSharpen == TRUE);

		__UpdateUI();
	}
	catch (CGalaxyException &e)
	{
		m_bCheckSharpen = bSharppenOld;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}
	catch (std::exception &e)
	{
		m_bCheckSharpen = bSharppenOld;
		UpdateData(FALSE);
		MessageBox(e.what());
		return;  	
	}
}

void CGxImageProcessDlg::OnClose()
{
	// TODO: Add your message handler code here and/or call default
	try
	{
		// Stop timer
		KillTimer(0);

		// Lose focus
		SetFocus();

		// Check whether the device has been stoped acquisition
		if (m_bIsSnap)
		{
			// Send AcquisitionStop command
			m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();

			// Stop Stream acquisition
			m_objStreamPtr->StopGrab();

			// Unregister the CaptureCallback function
			m_objStreamPtr->UnregisterCaptureCallback();

			m_bIsSnap = false;

		}
	}
	catch (CGalaxyException)
	{
		//do noting
	}
	catch (std::exception)
	{
		//do noting
	}

	try
	{
		// Check whether the device has been closed
		if (m_bIsOpen)
		{
			// Close stream
			m_objStreamPtr->Close();

			// Close device
			m_objDevicePtr->Close();

			m_bIsOpen = false;
		}
	}
	catch (CGalaxyException)
	{
		//do noting
	}
	catch (std::exception)
	{
		//do noting
	}

	try
	{
		//Release GxIAPICPP
		IGXFactory::GetInstance().Uninit();

	}
	catch (CGalaxyException)
	{
		//do noting
	}
	catch (std::exception)
	{
		//do noting
	}

	if (NULL != m_pBitmap)
	{
		delete m_pBitmap;
		m_pBitmap = NULL;
	}

	if (NULL != m_pSampleCaptureEventHandler)
	{
		delete m_pSampleCaptureEventHandler;
		m_pSampleCaptureEventHandler = NULL;
	}

	CDialog::OnClose();	
}

void CGxImageProcessDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	// TODO: Add your message handler code here and/or call default
	try
	{
		char PutText[16] = {0};

		if (pScrollBar == (CScrollBar*)&m_sliderLightness)           // Brightness Slider     
		{
			sprintf_s(PutText,sizeof(PutText), "%d", m_sliderLightness.GetPos());
			GetDlgItem(IDC_EDIT_LIGHTNESS)->SetWindowText(PutText);
			m_objImageProcessPtr->SetLightnessParam(m_sliderLightness.GetPos());	
		}
		else if (pScrollBar == (CScrollBar*)&m_sliderContrast)    // Contrast Slider
		{
			sprintf_s(PutText,sizeof(PutText), "%d", m_sliderContrast.GetPos());
			GetDlgItem(IDC_EDIT_CONTRAST)->SetWindowText(PutText);
			m_objImageProcessPtr->SetContrastParam(m_sliderContrast.GetPos());
		}
		else if (pScrollBar == (CScrollBar*)&m_sliderSaturation)      //Saturation Slider
		{
			sprintf_s(PutText,sizeof(PutText), "%d", m_sliderSaturation.GetPos());
			GetDlgItem(IDC_EDIT_SATURATION)->SetWindowText(PutText);
			m_objImageProcessPtr->SetSaturationParam(m_sliderSaturation.GetPos());
		}
		else if (pScrollBar == (CScrollBar*)&m_sliderGamma)      //Gamma Slider
		{
			sprintf_s(PutText,sizeof(PutText), "%f", (m_sliderGamma.GetPos()/PARAM_TRANSFORMATION_VALUE));
			GetDlgItem(IDC_EDIT_GAMMA)->SetWindowText(PutText);
			m_objImageProcessPtr->SetGammaParam(m_sliderGamma.GetPos()/PARAM_TRANSFORMATION_VALUE);
		}
		else if (pScrollBar == (CScrollBar*)&m_sliderSharpen)    //Sharpen Slider
		{
			sprintf_s(PutText,sizeof(PutText), "%f", (m_sliderSharpen.GetPos()/PARAM_TRANSFORMATION_VALUE));
			GetDlgItem(IDC_EDIT_SHARPEN)->SetWindowText(PutText);
			m_objImageProcessPtr->SetSharpenParam(m_sliderSharpen.GetPos()/PARAM_TRANSFORMATION_VALUE);
		}

		m_nOldSliderLightnessValue   = m_sliderLightness.GetPos();
		m_nOldSliderContrastValue    = m_sliderContrast.GetPos();
		m_nOldSliderSaturationValue  = m_sliderSaturation.GetPos();
		m_nOldSliderGammaValue       = m_sliderGamma.GetPos();
		m_nOldSliderSharpenValue     = m_sliderSharpen.GetPos();

	}
	catch (CGalaxyException &e)
	{
		m_sliderSharpen.SetPos(m_nOldSliderSharpenValue);
		m_sliderGamma.SetPos(m_nOldSliderGammaValue);
		m_sliderLightness.SetPos(m_nOldSliderLightnessValue);
		m_sliderContrast.SetPos(m_nOldSliderContrastValue);
		m_sliderSaturation.SetPos(m_nOldSliderSaturationValue);
		MessageBox(e.what());
		return;	
	}
	catch (std::exception &e)
	{
		m_sliderSharpen.SetPos(m_nOldSliderSharpenValue);
		m_sliderGamma.SetPos(m_nOldSliderGammaValue);
		m_sliderLightness.SetPos(m_nOldSliderLightnessValue);
		m_sliderContrast.SetPos(m_nOldSliderContrastValue);
		m_sliderSaturation.SetPos(m_nOldSliderSaturationValue);
		MessageBox(e.what());
		return;	
	}
	
	CDialog::OnHScroll(nSBCode, nPos, pScrollBar);
}

//---------------------------------------------------------------------
/**
\brief  This function demonstrates how to process image

\param  objImageDataPointer     image information   

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::ShowImageProcess(CImageDataPointer& objImageDataPointer)
{
	// The image processing parameters are first configured to avoid time consuming image processing
	m_objImageProcessPtr->SetValidBit(m_pBitmap->GetBestValudBit(objImageDataPointer->GetPixelFormat()));

	// Improve image quality
	m_pBitmap->ShowImageProcess(m_objImageProcessPtr,objImageDataPointer);
}

//---------------------------------------------------------------------
/**
\brief  This function demonstrates how to Change the width of the ComboBox drop-down box

\param  pComBox     the handle for combobox   

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::__ChangeComBoxWidth(CComboBox* pComBox)
{
	int nNumEntries = pComBox->GetCount();
	int nWidth = 0;
	CString str;

	CClientDC dc(pComBox);
	int nSave = dc.SaveDC();
	dc.SelectObject(GetFont());

	int nScrollWidth = ::GetSystemMetrics(SM_CXVSCROLL);
	for (int i = 0; i < nNumEntries; i++)
	{
		pComBox->GetLBText(i, str);
		int nLength = dc.GetTextExtent(str).cx + nScrollWidth;
		nWidth = max(nWidth, nLength);
	}

	// Add margin space to the calculations
	nWidth += dc.GetTextExtent("0").cx;

	dc.RestoreDC(nSave);
	pComBox->SetDroppedWidth(nWidth);

}

void CGxImageProcessDlg::OnCbnDropdownComboAwb()
{
	// TODO: Add your control notification handler code here

	// Change the width of the ComboBox drop-down box
	__ChangeComBoxWidth((CComboBox*)GetDlgItem(IDC_COMBO_AWB));
}

void CGxImageProcessDlg::OnCbnDropdownComboAwbLight()
{
	// TODO: Add your control notification handler code here

	// Change the width of the ComboBox drop-down box
	__ChangeComBoxWidth((CComboBox*)GetDlgItem(IDC_COMBO_AWB_LIGHT));
}

void CGxImageProcessDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default
	try
	{
		CString strText    = "";
		uint32_t i          = 0;
		CComboBox* pComBox = (CComboBox*)GetDlgItem(IDC_COMBO_AWB);

		// If the BalanceWhiteAuto mode is Once,After the successful setting, the actual BalanceWhiteAuto mode will be automatically turned off
		// So  once set to Once, the control is updated to off
		// Get the current BalanceWhiteAuto value from the device and then update to UI
		if (m_strBalanceWhiteAutoValue == "Once")
		{
			//Get the enumeration entry of BalanceWhiteAuto
			m_strBalanceWhiteAutoValue = m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->GetValue().c_str();
			GxIAPICPP::gxstring_vector vecBalanceWhiteAutoEnumCount;
			vecBalanceWhiteAutoEnumCount = m_objFeatureControlPtr->GetEnumFeature("BalanceWhiteAuto")->GetEnumEntryList();

			//Check whether the BalanceWhiteAuto is OFF
			if (m_strBalanceWhiteAutoValue == "Off")
			{
				for (i = 0; i< vecBalanceWhiteAutoEnumCount.size(); i++)
				{
					pComBox->GetLBText(i,strText);
					if (strText == "Off")
					{
						pComBox->SetCurSel(i);
						break;
					}
				}
			}
		}

	}
	catch (CGalaxyException)
	{
		// do nothing
	}
	catch (std::exception)
	{
		// do nothing
	}
	
	CDialog::OnTimer(nIDEvent);
}

BOOL CGxImageProcessDlg::PreTranslateMessage(MSG* pMsg)
{
	CWnd  *pWnd   = NULL;
	int   nCtrlID = 0;             

	// Check whether the keyboard enter the message
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN))   
	{    
		pWnd = GetFocus();

		// Get current control focus
		nCtrlID = pWnd->GetDlgCtrlID();

		// Lose focus
		SetFocus();

		return TRUE;
	}   
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_ESCAPE))  
	{   
		return  TRUE; 
	}   

	return CDialog::PreTranslateMessage(pMsg);
}

//---------------------------------------------------------------------
/**
\brief  Determine whether the camera turns on color correction

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::__IsEnableDeviceColorCorrection()
{
    m_bEnableColorCorrect = false;
    bool bIsImplement = m_objFeatureControlPtr->IsImplemented("ColorTransformationEnable");
    if(false == bIsImplement)
    {
        return;
    }

    CBoolFeaturePointer pBoolFeaturePtr = m_objFeatureControlPtr->GetBoolFeature("ColorTransformationEnable");
    m_bEnableColorCorrect = pBoolFeaturePtr->GetValue();
}

//---------------------------------------------------------------------
/**
\brief  Determine if the camera is enabled by Gamma enabling

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::__IsEnableDeviceGamma()
{
    m_bEnableGamma = false;
    bool bIsImplement = m_objFeatureControlPtr->IsImplemented("GammaEnable");
    if(false == bIsImplement)
    {
        return;
    }

    CBoolFeaturePointer pBoolFeaturePtr = m_objFeatureControlPtr->GetBoolFeature("GammaEnable");
    m_bEnableGamma = pBoolFeaturePtr->GetValue();
}

//---------------------------------------------------------------------
/**
\brief  Determine whether the camera turns on sharpening enable

\return void
*/
//---------------------------------------------------------------------
void CGxImageProcessDlg::__IsEnableDeviceSharpen()
{
    m_bEnableSharpness = false;
    bool bIsImplement = m_objFeatureControlPtr->IsImplemented("SharpnessMode");
    if(false == bIsImplement)
    {
        return;
    }

    CEnumFeaturePointer pEnumFeaturePtr = m_objFeatureControlPtr->GetEnumFeature("SharpnessMode");
    GxIAPICPP::gxstring strValue = pEnumFeaturePtr->GetValue();

    if(0 == strcmp(strValue, "Off"))
    {
        m_bEnableSharpness = false;
    }else if(0 == strcmp(strValue, "On"))
    {
        m_bEnableSharpness = true;
    }
}