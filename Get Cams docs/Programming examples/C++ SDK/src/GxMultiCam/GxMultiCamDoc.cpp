// GxMultiCamDoc.cpp : implementation of the CGxMultiCamDoc class
//

#include "stdafx.h"
#include "GxMultiCam.h"

#include "GxMultiCamDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CGxMuitCamDoc

IMPLEMENT_DYNCREATE(CGxMuitCamDoc, CDocument)

BEGIN_MESSAGE_MAP(CGxMuitCamDoc, CDocument)
END_MESSAGE_MAP()


// CGxMultiCamDoc construction/destruction

CGxMuitCamDoc::CGxMuitCamDoc()
{
	// TODO: add one-time construction code here

}

CGxMuitCamDoc::~CGxMuitCamDoc()
{
}

BOOL CGxMuitCamDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}




// CGxMultiCamDoc serialization

void CGxMuitCamDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add storing code here
	}
}


// CGxMultiCamDoc diagnostics

#ifdef _DEBUG
void CGxMuitCamDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CGxMuitCamDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CGxMultiCamDoc commands
