#helpers for using Python to generate JavaScript functions
#Note: for faster loading in the browser, all extraneous characters (i.e. space, tab, newline) removed


def showHist():
    return 'functionshowHist(hist_div){varconnected_div=document.getElementById(hist_div).style.display;varvis="block";if(connected_div=="none"){vis="block";}if(connected_div=="block"){vis="none";}document.getElementById(hist_div).style.display=vis;}'


def accordion():
    return 'varacc=document.getElementsByClassName("accordion");vari;for(i=0;i<acc.length;i++){acc[i].onclick=function(){this.classList.toggle("active");varpanel=this.nextElementSibling;if(panel.style.maxHeight){panel.style.maxHeight=null;}else{panel.style.maxHeight=panel.scrollHeight+"px";}}}'

def def_modalFunction():
    return '//GetthemodalfunctiongetModal(modalId,imageId,modalImageId,captionId){varmodal=document.getElementById(modalId);varimg=document.getElementById(imageId);varmodalImg=document.getElementById(modalImageId);varcaptionText=document.getElementById(captionId);img.onclick=function(){modal.style.display="block";modalImg.src=this.src;captionText.innerHTML=this.alt;}varspan=document.getElementsByClassName("close")[0];span.onclick=function(){modal.style.display="none";}}'                        

def call_modalFunction(modalId,imageId,modalImageId,captionId):
    return 'getModal("{modalId}","{imageId}","{modalImageId}","{captionId}")'.format(modalId=modalId,
                                                                                     imageId=imageId,
                                                                                     modalImageId=modalImageId,
                                                                                     captionId=captionId)

def uncheckAll():
    return 'varcheckboxes=document.getElementsByTagName("input");for(vari=0;i<checkboxes.length;i++){if(checkboxes[i].type=="checkbox"){checkboxes[i].checked=false;}}'


def selectClusterTab():
    return 'functionopenCluster(evt,cityName){//Declareallvariablesvari,tabcontent,tablinks;//Getallelementswithclass="tabcontent"andhidethemtabcontent=document.getElementsByClassName("tabcontent");for(i=0;i<tabcontent.length;i++){tabcontent[i].style.display="none";}//Getallelementswithclass="tablinks"andremovetheclass"active"tablinks=document.getElementsByClassName("tablinks");for(i=0;i<tablinks.length;i++){tablinks[i].className=tablinks[i].className.replace("active","");}//Showthecurrenttab,andaddan"active"classtothebuttonthatopenedthetabdocument.getElementById(cityName).style.display="block";evt.currentTarget.className+="active";}'
