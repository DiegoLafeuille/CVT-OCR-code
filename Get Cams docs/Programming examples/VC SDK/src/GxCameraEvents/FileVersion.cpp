#include "stdafx.h"
#include "FileVersion.h"

CFileVersion::CFileVersion(CString strPath)
{
	m_strPath = strPath;
}

CFileVersion::~CFileVersion()
{
}


CString CFileVersion::GetFullVersion()
{
	CString strRet;
	strRet = "FileVersion: " + GetFileVersion() + "\n"; 
	strRet += "Description: " + GetFileDescription() +"\n";
    strRet += "Copyright: " + GetLegalCopyright() +"\n";
    strRet += "Comments: " + GetComments() + "\n";
    strRet += "ProductVersion: " + GetProductVersion() +"\n";
	strRet += "ProductName: " + GetProductName() +"\n";
	strRet += "PrivateBuild Description: " + GetPrivateBuild() +"\n";
	strRet += "CompanyName: " + GetCompanyName() +"\n";
	strRet += "LegalTrademarks: " + GetLegalTrademarks() +"\n";
	strRet += "InternalName: " + GetInternalName() +"\n";
	strRet += "SpecialBuild Description: " + GetSpecialBuild() +"\n";
	strRet += "OriginalFilename: " + GetOriginalFilename() +"\n";	
	return strRet;
}


CString CFileVersion::GetComments()
{
	return GetVer("Comments");
}


CString CFileVersion::GetInternalName()
{
	return GetVer("InternalName");
}


CString CFileVersion::GetProductName()
{
	return GetVer("ProductName");
}


CString CFileVersion::GetCompanyName()
{
	return GetVer("CompanyName");
}


CString CFileVersion::GetLegalCopyright()
{
	return GetVer("LegalCopyright");
}


CString CFileVersion::GetProductVersion()
{
	return GetVer("ProductVersion");
}


CString CFileVersion::GetFileDescription()
{
	return GetVer("FileDescription");
}


CString CFileVersion::GetLegalTrademarks()
{
	return GetVer("LegalTrademarks");
}

CString CFileVersion::GetPrivateBuild()
{
	return GetVer("PrivateBuild");
}


CString CFileVersion::GetFileVersion()
{
	return GetVer("FileVersion");
}

CString CFileVersion::GetOriginalFilename()
{
	return GetVer("OriginalFilename");
}

CString CFileVersion::GetSpecialBuild()
{
	return GetVer("SpecialBuild");
}


CString CFileVersion::GetVer(CString strSubBlock)
{
	struct LANGANDCODEPAGE 
	{
		WORD wLanguage;
		WORD wCodePage;
	}*lpTranslate;
	
	int     i             = 0;
	CString strRet        = "";
	DWORD   dwHandle      = 0;
	DWORD   dwInfoSize    = 0;
	UINT    cbTranslate   = 0;
	UINT    dwBytes       = 0;
	char    chBlock[256]  = {0};
	char*   pBuffer       = NULL;
	char*   pData         = NULL;

	dwInfoSize = GetFileVersionInfoSize((char*)(LPCTSTR)m_strPath, &dwHandle);

	if(dwInfoSize != 0)
	{
		pData = new char[dwInfoSize];
		if( GetFileVersionInfo((char*)(LPCTSTR)m_strPath, 0, dwInfoSize, pData))
		{
			
			VerQueryValue(pData, TEXT("\\VarFileInfo\\Translation"), (LPVOID*)&lpTranslate, &cbTranslate);

			for(i = 0; i <(int)(cbTranslate/sizeof(struct LANGANDCODEPAGE)); i++)
			{
				wsprintf(chBlock, 
					     TEXT("\\StringFileInfo\\%04x%04x\\%s"),
					     lpTranslate[i].wLanguage,
					     lpTranslate[i].wCodePage,
					     strSubBlock);		
				
				VerQueryValue(pData, 
					         chBlock, 
					         (LPVOID*)&pBuffer, 
				          	 &dwBytes); 
				
				strRet.Format("%s", pBuffer);

				break; //only retrieve the first language and code page
			}
		}
		if (pData != NULL)
		{
			delete []pData;
			pData = NULL;
		}
	}

	return strRet;
}