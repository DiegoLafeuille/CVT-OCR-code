// SltCamDlg.cpp : implementation file
//

#include "stdafx.h"
#include "GxMultiCam.h"
#include "SltCamDlg.h"
#include "GxEx.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CSltCamDlg dialog


CSltCamDlg::CSltCamDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CSltCamDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CSltCamDlg)
	//}}AFX_DATA_INIT
    m_pMainFrame = (CMainFrame*)AfxGetMainWnd();
	m_nDeviceNum = 0;
}


void CSltCamDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CSltCamDlg)
	DDX_Control(pDX, IDC_CMB_CAM, m_cmbcam);
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CSltCamDlg, CDialog)
	//{{AFX_MSG_MAP(CSltCamDlg)
	ON_CBN_EDITCHANGE(IDC_CMB_CAM, OnEditchangeCmbCam)
	ON_BN_CLICKED(IDC_BTN_REENUM, OnBnClickedBtnEnum)
	//}}AFX_MSG_MAP 
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CSltCamDlg message handlers

BOOL CSltCamDlg::OnInitDialog() 
{
	CDialog::OnInitDialog();
	
	// TODO: Add extra initialization here
	uint32_t  nDevNum = m_pMainFrame->m_nCamNum;
	uint32_t  nMinDevMum = 0;
	
	if(nDevNum > 0)
	{ 
		// The device list supports 4 devices at most
		nMinDevMum = min(nDevNum, MAX_DEVICE_MUN);

		// Show the device information to the list if the device number is not 0.
		m_cmbcam.ResetContent();
		for(uint32_t i = 0; i < nMinDevMum; i++)
		{
			//Show the name of the running device 
			m_cmbcam.InsertString(-1, m_pMainFrame->m_pBaseinfo[i].szDisplayName);	
		}

		// Update the current operating device to the device list.
		m_cmbcam.SetCurSel(m_pMainFrame->m_nOperateID);
	}
	else
	{
		// If the connected device is 0 then enumerate device again.
        UpdateDeviceList();

        if (AllocBufferForMainFrame())
        {
			// Show the device information to the list if the device number is not 0.
			m_cmbcam.ResetContent();
			for(uint32_t i = 0; i < m_nDeviceNum; i++)
			{
				//Show the name of the running device 
				m_cmbcam.InsertString(-1, m_pMainFrame->m_pBaseinfo[i].szDisplayName);	
			}
			m_cmbcam.SetCurSel(0);

			m_pMainFrame->m_nCamNum = m_nDeviceNum;
        }
		else
		{
			m_pMainFrame->m_nCamNum = 0;
			MessageBox("Allocate memory failed!");		
			return FALSE;
		}
	}

	return TRUE;  // return TRUE unless you set the focus to a control
	              // EXCEPTION: OCX Property Pages should return FALSE
}

void CSltCamDlg::OnEditchangeCmbCam() 
{
	// TODO: Add your control notification handler code here
	m_cmbcam.SetCurSel(0);
}

void CSltCamDlg::OnOK() 
{
	// TODO: Add extra validation here
	// Get the index of selection
	m_pMainFrame->m_nOperateID = m_cmbcam.GetCurSel();

	// Check whether the index is exist
	if (m_pMainFrame->m_nOperateID >= 0)
	{
		//Update the device name to the headline of the main window 
		m_pMainFrame->SetWindowText(m_pMainFrame->m_pBaseinfo[m_pMainFrame->m_nOperateID].szDisplayName);
	}
	else
	{
        m_pMainFrame->SetWindowText("GxMutiCam.exe");
	}
	CDialog::OnOK();
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for the main window

\return Success: TRUE     Fail:FALSE   
*/
//----------------------------------------------------------------------------------
bool CSltCamDlg::AllocBufferForMainFrame()
{
	uint32_t i = 0;

	//-------------------Allocate memory------------------------
	m_pMainFrame->m_hDevices = new GX_DEV_HANDLE[m_nDeviceNum];
	if (m_pMainFrame->m_hDevices == NULL)
	{
		ClearBuffer();
		return false;
	}

	m_pMainFrame->m_pstCam=new CAMER_INFO[m_nDeviceNum];
	if (m_pMainFrame->m_pstCam == NULL)
	{
		ClearBuffer();
		return false;
	}

	m_pMainFrame->m_pCamsFps = new CFps[m_nDeviceNum];
	if (m_pMainFrame->m_pCamsFps == NULL)
	{
		ClearBuffer();
		return false;
	}

	//--------------Initialize and allocate memory--------------------
	for (i=0; i<m_nDeviceNum; i++)
	{
		m_pMainFrame->m_hDevices[i]  = NULL;

		m_pMainFrame->m_pstCam[i].bIsColorFilter = false;
		m_pMainFrame->m_pstCam[i].bIsOpen        = FALSE;
		m_pMainFrame->m_pstCam[i].bIsSnap        = FALSE;
		m_pMainFrame->m_pstCam[i].pBmpInfo       = NULL;
		m_pMainFrame->m_pstCam[i].pRawBuffer     = NULL;
		m_pMainFrame->m_pstCam[i].pImageBuffer   = NULL;
		m_pMainFrame->m_pstCam[i].fFps           = 0.0;
        m_pMainFrame->m_pstCam[i].nBayerLayout   = 0;
		m_pMainFrame->m_pstCam[i].nImageHeight   = 0;
		m_pMainFrame->m_pstCam[i].nImageWidth    = 0;
		m_pMainFrame->m_pstCam[i].nPayLoadSise   = 0;
		memset(m_pMainFrame->m_pstCam[i].chBmpBuf,0,sizeof(m_pMainFrame->m_pstCam[i].chBmpBuf));
	}

	return true;
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to enumerate all camera devices and initialize device.

\return void
*/
//----------------------------------------------------------------------------------
void CSltCamDlg::UpdateDeviceList()
{
	GX_STATUS  emStatus   = GX_STATUS_SUCCESS;
    uint32_t   nDeviceNum = 0;
	bool       bRet       = true;
	size_t     nSize      = 0;

	// Enumerate all camera devices 
	emStatus = GXUpdateDeviceList(&nDeviceNum, 1000);
	if (emStatus != GX_STATUS_SUCCESS)
	{
		ShowErrorString(emStatus);
		return;
	}

	
	if (nDeviceNum > 0)
	{
		// Allocate memory for getting device information 
		m_pMainFrame->m_pBaseinfo = new GX_DEVICE_BASE_INFO[nDeviceNum];
		if (m_pMainFrame->m_pBaseinfo == NULL)
		{
			MessageBox("Fail to allocate memory for device");

			nDeviceNum = 0;

			return;
		}

		//Get all the information of devices
		nSize = nDeviceNum * sizeof(GX_DEVICE_BASE_INFO);  
		emStatus = GXGetAllDeviceBaseInfo(m_pMainFrame->m_pBaseinfo, &nSize);
		if (emStatus != GX_STATUS_SUCCESS)
		{
			ShowErrorString(emStatus);
			delete []m_pMainFrame->m_pBaseinfo;
			m_pMainFrame->m_pBaseinfo = NULL;

			nDeviceNum = 0;

			return;
		}
	}

	m_nDeviceNum = min(nDeviceNum, MAX_DEVICE_MUN);
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to allocate memory for enumerating devices

\return void
*/
//----------------------------------------------------------------------------------
void CSltCamDlg::OnBnClickedBtnEnum() 
{
	// TODO: Add your control notification handler code here
	GX_STATUS emStatus = GX_STATUS_SUCCESS;

	// Stop showing the information of devices 
    m_pMainFrame->m_nOperateID = -1;

	// If the device is acquiring image then stop it.
	emStatus = CloseCamer();
	if (emStatus != GX_STATUS_SUCCESS)
	{
		// TODO: Add your error handler code here  
	}

	m_cmbcam.ResetContent();
	ClearBuffer();

	UpdateDeviceList();

	if (m_nDeviceNum <= 0)
	{
		m_pMainFrame->m_nCamNum = m_nDeviceNum;
		return ;
	}

	// Allocate memory for the main window
	if (AllocBufferForMainFrame())
	{
		// The device list supports 4 devices at most
		for(uint32_t i = 0; i < m_nDeviceNum; i++)
		{
			//Show the name of the running device 
			m_cmbcam.InsertString(-1, m_pMainFrame->m_pBaseinfo[i].szDisplayName);	
		}
		m_cmbcam.SetCurSel(0);
		m_pMainFrame->m_nOperateID = 0;
		m_pMainFrame->m_nCamNum = m_nDeviceNum;
	}
	else
	{
		m_pMainFrame->m_nOperateID = -1;
		m_pMainFrame->m_nCamNum = 0;

		MessageBox("Allocate memory failed!");
		return;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to release memory for saving device information

\return void
*/
//----------------------------------------------------------------------------------
void CSltCamDlg::ClearBuffer()
{
	CMainFrame* pFrame =(CMainFrame*)AfxGetMainWnd();
	CAMER_INFO *pTmpInfo = NULL;

	// Release memory for acquiring image
	if (pFrame->m_pstCam != NULL)
	{
		for (uint32_t i = 0; i < pFrame->m_nCamNum; i++)
		{
			if(pFrame->m_pstCam[i].pImageBuffer != NULL)
			{
				delete[]pFrame->m_pstCam[i].pImageBuffer;
				pFrame->m_pstCam[i].pImageBuffer = NULL;
			}

			if (pFrame->m_pstCam[i].pRawBuffer != NULL)
			{
				delete []pFrame->m_pstCam[i].pRawBuffer;
				pFrame->m_pstCam[i].pRawBuffer = NULL;
			}

			::ReleaseDC(pFrame->m_pCWnds[i]->m_hWnd,pFrame->m_hDCs[i]);
			pFrame->m_hDCs[i] = NULL;
		}

		delete[]pFrame->m_pstCam;
		pFrame->m_pstCam = NULL;
	}

	// Clear the device handle
	if(pFrame->m_hDevices != NULL)
	{
		delete[]pFrame->m_hDevices;
		pFrame->m_hDevices = NULL;
	}

	// Clear the device information
	if(pFrame->m_pBaseinfo != NULL)
	{
		delete[]pFrame->m_pBaseinfo;
		pFrame->m_pBaseinfo = NULL;
	}

	// Release the pointer of fps 
	if (pFrame->m_pCamsFps != NULL)
	{
		delete[] pFrame->m_pCamsFps;
		pFrame->m_pCamsFps = NULL;
	}
}

//----------------------------------------------------------------------------------
/**
\brief  This function demonstrates how to convert the error code to error description
\param  emErrorStatus  [in]
\return void
*/
//----------------------------------------------------------------------------------
void CSltCamDlg::ShowErrorString(GX_STATUS emErrorStatus)
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
\brief  This function demonstrates how to stop acquiring image and close all the devices.

\return emStatus GX_STATUS_SUCCESS: Success Else: Fail
*/
//----------------------------------------------------------------------------------
GX_STATUS CSltCamDlg::CloseCamer()
{
	GX_STATUS emStatus = GX_STATUS_SUCCESS;
	uint32_t  i = 0;


 
	for(i = 0; i < (uint32_t)m_pMainFrame->m_nCamNum; i++)
	{
		if(m_pMainFrame->m_pstCam[i].bIsSnap)
		{
			//If the device is acquiring image then stop it.
			emStatus = GXSendCommand(m_pMainFrame->m_hDevices[i], GX_COMMAND_ACQUISITION_STOP);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				// TODO: Add your error handler code here  
			}
			
			//Unregister the CaptureCallback function 
			emStatus = GXUnregisterCaptureCallback(m_pMainFrame->m_hDevices[i]);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				// TODO: Add your error handler code here  
			}
			
			m_pMainFrame->m_pstCam[i].bIsSnap = FALSE;
		}
		if(m_pMainFrame->m_pstCam[i].bIsOpen)
		{
			emStatus = GXCloseDevice(m_pMainFrame->m_hDevices[i]);
			if (emStatus != GX_STATUS_SUCCESS)
			{
				// TODO: Add your error handler code here  
			}
			
			m_pMainFrame->m_hDevices[i] = NULL;
			m_pMainFrame->m_pstCam[i].bIsOpen = FALSE;
		}
	}

	return emStatus;
}

