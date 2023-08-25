#include "stdafx.h"
#include "FileVersion.h"

CFileVersion::CFileVersion(CString szPath)
{
	m_szPath = szPath;
}

CFileVersion::~CFileVersion()
{
}

CString CFileVersion::GetFullVersion()
{
	CString szRet;
	szRet = "FileVersion: " + GetFileVersion() + "\n"; 
	szRet += "Description: " + GetFileDescription() +"\n";
	szRet += "Copyright: " + GetLegalCopyright() +"\n";
	szRet += "Comments: " + GetComments() + "\n";
	szRet += "ProductVersion: " + GetProductVersion() +"\n";
	szRet += "ProductName: " + GetProductName() +"\n";
	szRet += "PrivateBuild Description: " + GetPrivateBuild() +"\n";
	szRet += "CompanyName: " + GetCompanyName() +"\n";
	szRet += "LegalTrademarks: " + GetLegalTrademarks() +"\n";
	szRet += "InternalName: " + GetInternalName() +"\n";
	szRet += "SpecialBuild Description: " + GetSpecialBuild() +"\n";
	szRet += "OriginalFilename: " + GetOriginalFilename() +"\n";
	return szRet;
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

CString CFileVersion::GetVer(CString szSubBlock)
{
	CString szRet;

	struct LANGANDCODEPAGE {
		WORD wLanguage;
		WORD wCodePage;
	} *lpTranslate;

	DWORD dwHandle;
	DWORD dwInfoSize;

	dwInfoSize = GetFileVersionInfoSize((char*)(LPCTSTR)m_szPath, &dwHandle);

	if( dwInfoSize != 0 )
	{
		char* lpData = new char[dwInfoSize];
		if( GetFileVersionInfo((char*)(LPCTSTR)m_szPath, 0, dwInfoSize, lpData) )
		{
			UINT cbTranslate = 0;
			VerQueryValue(lpData, 
              TEXT("\\VarFileInfo\\Translation"),
              (LPVOID*)&lpTranslate,
              &cbTranslate);

			char SubBlock[256];
			char* lpBuffer;
			UINT dwBytes = 0;
			for(int  i=0; i <(int) (cbTranslate/sizeof(struct LANGANDCODEPAGE)); i++ )
			{
				wsprintf( SubBlock, 
					TEXT("\\StringFileInfo\\%04x%04x\\%s"),
					lpTranslate[i].wLanguage,
					lpTranslate[i].wCodePage,
					szSubBlock);		
				
				VerQueryValue(lpData, 
					SubBlock, 
					(LPVOID*)&lpBuffer, 
					&dwBytes); 
				
				szRet.Format("%s", lpBuffer);

				break; //only retrieve the first language and code page
			}

		}
		delete [] lpData;
		lpData = NULL;
	}

	return szRet;
}