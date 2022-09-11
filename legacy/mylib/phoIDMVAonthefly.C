#include <iostream>
using namespace std;


struct phoIDInput{
	float phoIdMva_SCRawE_;
	float phoIdMva_R9_;
	float phoIdMva_covIEtaIEta_;
	float phoIdMva_PhiWidth_;
	float phoIdMva_EtaWidth_;
	float phoIdMva_covIEtaIPhi_;
	float phoIdMva_S4_;
	float phoIdMva_pfPhoIso03_;
	float phoIdMva_pfChgIso03_;
	float phoIdMva_pfChgIso03worst_;
	float phoIdMva_ScEta_;
	float phoIdMva_rho_;
	float phoIdMva_ESEffSigmaRR_;
	float phoIdMva_esEnovSCRawEn_;
};



TMVA::Reader* bookReadersEB(const string &xmlfilenameEB, phoIDInput &inp){
	// **** bdt 2015 EB ****
	//cout << "inside" << endl;

	string mvamethod = "BDT";

	TMVA::Reader* phoIdMva_EB_ = new TMVA::Reader( "!Color:Silent" );

	phoIdMva_EB_->AddVariable( "SCRawE", 			&inp.phoIdMva_SCRawE_ );
	phoIdMva_EB_->AddVariable( "r9", 			&inp.phoIdMva_R9_ );
	phoIdMva_EB_->AddVariable( "sigmaIetaIeta", 		&inp.phoIdMva_covIEtaIEta_ );
	phoIdMva_EB_->AddVariable( "etaWidth", 			&inp.phoIdMva_EtaWidth_ );
	phoIdMva_EB_->AddVariable( "phiWidth", 			&inp.phoIdMva_PhiWidth_ );
	phoIdMva_EB_->AddVariable( "covIEtaIPhi", 		&inp.phoIdMva_covIEtaIPhi_ );
	phoIdMva_EB_->AddVariable( "s4", 			&inp.phoIdMva_S4_ );
	phoIdMva_EB_->AddVariable( "phoIso03", 			&inp.phoIdMva_pfPhoIso03_ );
	phoIdMva_EB_->AddVariable( "chgIsoWrtChosenVtx", 	&inp.phoIdMva_pfChgIso03_ );
	phoIdMva_EB_->AddVariable( "chgIsoWrtWorstVtx", 	&inp.phoIdMva_pfChgIso03worst_ );
	phoIdMva_EB_->AddVariable( "scEta", 			&inp.phoIdMva_ScEta_ );
	phoIdMva_EB_->AddVariable( "rho", 			&inp.phoIdMva_rho_ );
	phoIdMva_EB_->BookMVA( mvamethod.c_str(), xmlfilenameEB );

	return phoIdMva_EB_;
}



TMVA::Reader* bookReadersEE(const string &xmlfilenameEE, phoIDInput &inp, bool rhoCorr=false, bool leg2016=false){
	//cout << "inside" << endl;
	// **** bdt 2015 EE ****

	string mvamethod = "BDT";

	TMVA::Reader* phoIdMva_EE_ = new TMVA::Reader( "!Color:Silent" );

	phoIdMva_EE_->AddVariable( "SCRawE", 		&inp.phoIdMva_SCRawE_ );
	phoIdMva_EE_->AddVariable( "r9", 		&inp.phoIdMva_R9_ );
	phoIdMva_EE_->AddVariable( "sigmaIetaIeta", 	&inp.phoIdMva_covIEtaIEta_ );
	phoIdMva_EE_->AddVariable( "etaWidth", 		&inp.phoIdMva_EtaWidth_ );
	phoIdMva_EE_->AddVariable( "phiWidth", 		&inp.phoIdMva_PhiWidth_ );
	phoIdMva_EE_->AddVariable( "covIEtaIPhi", 	&inp.phoIdMva_covIEtaIPhi_ );
	phoIdMva_EE_->AddVariable( "s4", 		&inp.phoIdMva_S4_ );
	//cout << "first step";
	if( leg2016 ) 
		phoIdMva_EE_->AddVariable( "isoPhoCorrMax2p5", 	&inp.phoIdMva_pfPhoIso03_ );
	else 
		phoIdMva_EE_->AddVariable( "phoIso03", 		&inp.phoIdMva_pfPhoIso03_ );

	//cout << "passed";
	phoIdMva_EE_->AddVariable( "chgIsoWrtChosenVtx", 	&inp.phoIdMva_pfChgIso03_ );
	phoIdMva_EE_->AddVariable( "chgIsoWrtWorstVtx", 	&inp.phoIdMva_pfChgIso03worst_ );
	phoIdMva_EE_->AddVariable( "scEta", 			&inp.phoIdMva_ScEta_ );
	phoIdMva_EE_->AddVariable( "rho", 			&inp.phoIdMva_rho_ );
	phoIdMva_EE_->AddVariable( "esEffSigmaRR", 		&inp.phoIdMva_ESEffSigmaRR_ );
	//cout << "second step";
	// if(rhoCorr) phoIdMva_EE_->AddVariable( "esEnovSCRawEn", 	&inp.phoIdMva_esEnovSCRawEn_ );
	if( leg2016 ) 
		phoIdMva_EE_->AddVariable( "esEnergy/SCRawE", 	&inp.phoIdMva_esEnovSCRawEn_ );
	else 
		phoIdMva_EE_->AddVariable( "esEnergyOverRawE", 	&inp.phoIdMva_esEnovSCRawEn_ );
	phoIdMva_EE_->BookMVA( mvamethod.c_str(), xmlfilenameEE );
	//cout << "out";
	return phoIdMva_EE_;

}

