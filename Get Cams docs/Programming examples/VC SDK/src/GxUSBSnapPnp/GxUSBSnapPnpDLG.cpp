// GxUSBSnapPnpDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxUSBSnapPnp.h"
#include "GxUSBSnapPnpDlg.h"
#include <DBT.H>

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
// CGxUSBSnapPnpDlg dialog

CGxUSBSnapPnpDlg::CGxUSBSnapPnpDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CGxUSBSnapPnpDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CGxUSBSnapPnpDlg)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
	// Note that LoadIcon does not require a subsequent DestroyIcon in Win32
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CGxUSBSnapPnpDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGxUSBSnapPnpDlg)
	DDX_Control(pDX, IDC_LIST_RESULT, m_listResult);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CGxUSBSnapPnpDlg, CDialog)
	//{{AFX_MSG_MAP(CGxUSBSnapPnpDlg)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	//}}AFX_MSG_MAP
	ON_WM_CLOSE()
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGxUSBSnapPnpDlg message handlers

BOOL CGxUSBSnapPnpDlg::OnInitDialog()
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
    int i = 0;

	static GUID CameraGUIDs[] = 
	{	
		//GUID
		{0x889ff8a4, 0x40c8, 0x47bc, {0x99, 0x1e, 0x65, 0xc3, 0xb6, 0x83, 0x1f, 0x68}},
		{0xdbc348a2, 0xca28, 0x43c2, {0x8f, 0xdd, 0x69, 0xc3, 0x7e, 0xd2, 0x50, 0x28}}
	}; 

	// A loop to register every GUID of device 
	for(i = 0; i < 9; i++)
	{
		RegHVSnapPnpGUID(m_hWnd,CameraGUIDs[i]);
	}

	InitListHeader();

	GX_STATUS  emStatus = GX_STATUS_SUCCESS;
	/// Before using any GxIAPI methods, the GxIAPI must be initialized.  
	emStatus = GXInitLib();  
	if (emStatus != GX_STATUS_SUCCESS)
	{
        MessageBox("Fail to initialize GxIAPI !");
		exit(0);
	}

	//When the program start running,enumerate all the devices and update the device information to UI
	UpdateDeviceList();

	//Initialize the status of whether the device has been changed
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGxUSBSnapPnpDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

void CGxUSBSnapPnpDlg::OnPaint() 
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
HCURSOR CGxUSBSnapPnpDlg::OnQueryDragIcon()
{
	return (HCURSOR) m_hIcon;
}

BOOL CGxUSBSnapPnpDlg::PreTranslateMessage(MSG* pMsg) 
{
	// TODO: Add your specialized code here and/or call the base class
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_RETURN)) 
	{		
		return TRUE;
	}   
	if ((pMsg->message == WM_KEYDOWN) && (pMsg->wParam == VK_ESCAPE))  
	{   
		return   TRUE; 
	}
	return CDialog::PreTranslateMessage(pMsg);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to initialize the ListHeader 

\return void
*/
//----------------------------------------------------------------------------------
void CGxUSBSnapPnpDlg::InitListHeader()
{
	//Set style of the table
	m_listResult.SetExtendedStyle(LVS_EX_INFOTIP | LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
	
	//Add the ListHeader
	m_listResult.InsertColumn(0 , "Index", LVCFMT_LEFT, 110);
	m_listResult.InsertColumn(1 , "Name", LVCFMT_LEFT, 310);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all the devices and update device information of UI

\return void
*/
//----------------------------------------------------------------------------------
void CGxUSBSnapPnpDlg::UpdateDeviceList()
{
	GX_STATUS            emStatus     = GX_STATUS_ERROR;  
	size_t               nSize        = 0;
	uint32_t             nDeviceNum   = 0;             
	GX_DEVICE_BASE_INFO *pDeviceInfo  = NULL;              
	CString              strTemp      = "";

	m_listResult.DeleteAllItems();

	//Enumerate all camera devices
	emStatus = GXUpdateDeviceList(&nDeviceNum, 2000);	
	if (emStatus == GX_STATUS_SUCCESS)		
	{	
		if (nDeviceNum != 0)
		{
			// Allocate memory for saving device information
			pDeviceInfo = new GX_DEVICE_BASE_INFO[nDeviceNum];
			if (pDeviceInfo == NULL)
			{
                MessageBox("Fail to allocate memory !");
				return;
			}
			nSize = nDeviceNum * sizeof(GX_DEVICE_BASE_INFO);	
			
			emStatus = GXGetAllDeviceBaseInfo(pDeviceInfo, &nSize);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				if (pDeviceInfo != NULL)
				{
					delete[]pDeviceInfo;
					pDeviceInfo = NULL;
				}
                MessageBox("Fail to acquire device information!");
				return;
			}

			//Update the device information to UI
			GX_DEVICE_BASE_INFO *pTempBaseInfo = pDeviceInfo;
			for (int i = 0; i < (int)nDeviceNum; i++)
			{
				strTemp.Format("%d",i);
				m_listResult.InsertItem(i,strTemp);
				strTemp.Format("%s",pTempBaseInfo->szDisplayName);
				m_listResult.SetItemText(i,1,strTemp);
				pTempBaseInfo++;
			}

		}
	}
	else
	{
		MessageBox("Fail to acquire device information!");
		return;
	}

	// Release memory
	if (pDeviceInfo != NULL)
	{
		delete[]pDeviceInfo;
		pDeviceInfo = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to register the camera and return the status
\param  handle   [in]   
\param  stGuid   [in]  
\return false: Fail; ture:Success
*/
//----------------------------------------------------------------------------------
BOOL CGxUSBSnapPnpDlg::RegHVSnapPnpGUID(HANDLE handle, GUID stGuid)
{
	BOOL bReturn = FALSE;
	DEV_BROADCAST_DEVICEINTERFACE NotificationFilter;
	ZeroMemory(&NotificationFilter, sizeof(NotificationFilter));
	NotificationFilter.dbcc_size = sizeof(DEV_BROADCAST_DEVICEINTERFACE);
	NotificationFilter.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE;
	NotificationFilter.dbcc_classguid = stGuid; 
	HDEVNOTIFY hDevInterfaceNotify = ::RegisterDeviceNotification(handle, &NotificationFilter, DEVICE_NOTIFY_WINDOW_HANDLE);
	if (hDevInterfaceNotify)
	{
		bReturn = TRUE;
	}
	
	return bReturn;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to process PNP event

\return void
*/
//----------------------------------------------------------------------------------
LRESULT CGxUSBSnapPnpDlg::DefWindowProc(UINT message, WPARAM wParam, LPARAM lParam) 
{
	// TODO: Add your specialized code here and/or call the base class
	if(message == WM_DEVICECHANGE)
	{
		switch(wParam) 
		{
		case DBT_DEVICEARRIVAL:                  //A device is plugged in
			{  
				PDEV_BROADCAST_HDR  devHdr;
				devHdr = (PDEV_BROADCAST_HDR)lParam;

                 
				switch (devHdr->dbch_devicetype)
				{
				case DBT_DEVTYP_DEVICEINTERFACE:  //The device has been registered  
					{
						// Enumerate all the devices and update UI  
					    UpdateDeviceList();
					}
					break;
				default:
					break;
				}				
			}
			break;
			
		case DBT_DEVICEREMOVECOMPLETE:            // A device is pulled out           
			{
				PDEV_BROADCAST_HDR  devHdr;
				devHdr = (PDEV_BROADCAST_HDR)lParam;

				switch ( devHdr->dbch_devicetype )
				{
				case DBT_DEVTYP_DEVICEINTERFACE :  // The device has been registered
					{
						// Enumerate all the devices and update UI  
						UpdateDeviceList();
					}
					break;
				default:
					break;
				}
			}
			break;
			
		default:
			break;
		}	
	}
	return CDialog::DefWindowProc(message, wParam, lParam);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to close the program

\return void
*/
//----------------------------------------------------------------------------------
void CGxUSBSnapPnpDlg::OnClose()
{
	// TODO: Add your message handler code here and/or call default
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Release GxIAPI
	emStatus = GXCloseLib();

	CDialog::OnClose();
}
