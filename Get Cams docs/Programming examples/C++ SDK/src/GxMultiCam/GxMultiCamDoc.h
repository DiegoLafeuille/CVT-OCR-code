// GxMultiCamDoc.h : interface of the CGxMultiCamDoc class
//


#pragma once
#include "GalaxyIncludes.h"

class CGxMuitCamDoc : public CDocument
{
protected: // create from serialization only
	CGxMuitCamDoc();
	DECLARE_DYNCREATE(CGxMuitCamDoc)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);

// Implementation
public:
	virtual ~CGxMuitCamDoc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
};


