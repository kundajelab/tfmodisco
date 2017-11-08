#helpers for using Python to generate JavaScript functions
#Note: for faster loading in the browser, all extraneous characters (i.e. space, tab, newline) removed


def showHist():
    return 'functionshowHist(hist_div){varconnected_div=document.getElementById(hist_div).style.display;varvis="block";if(connected_div=="none"){vis="block";}if(connected_div=="block"){vis="none";}document.getElementById(hist_div).style.display=vis;}'


def accordion():
    return 'varacc=document.getElementsByClassName("accordion");vari;for(i=0;i<acc.length;i++){acc[i].onclick=function(){this.classList.toggle("active");varpanel=this.nextElementSibling;if(panel.style.maxHeight){panel.style.maxHeight=null;}else{panel.style.maxHeight=panel.scrollHeight+"px";}}}'

def zoomAllMetaClustersHeatmap():
    return '//Getthemodalvarmodal=document.getElementById("myModal");//Gettheimageandinsertitinsidethemodal-useits"alt"textasacaptionvarimg=document.getElementById("myImg");varmodalImg=document.getElementById("img01");varcaptionText=document.getElementById("caption");img.onclick=function(){modal.style.display="block";modalImg.src=this.src;captionText.innerHTML=this.alt;}//Getthe<span>elementthatclosesthemodalvarspan=document.getElementsByClassName("close")[0];//Whentheuserclickson<span>(x),closethemodalspan.onclick=function(){modal.style.display="none";}'

def uncheckAll():
    return 'varcheckboxes=document.getElementsByTagName("input");for(vari=0;i<checkboxes.length;i++){if(checkboxes[i].type=="checkbox"){checkboxes[i].checked=false;}}</script>'


def selectClusterTab():
    return 'functionopenCluster(evt,cityName){//Declareallvariablesvari,tabcontent,tablinks;//Getallelementswithclass="tabcontent"andhidethemtabcontent=document.getElementsByClassName("tabcontent");for(i=0;i<tabcontent.length;i++){tabcontent[i].style.display="none";}//Getallelementswithclass="tablinks"andremovetheclass"active"tablinks=document.getElementsByClassName("tablinks");for(i=0;i<tablinks.length;i++){tablinks[i].className=tablinks[i].className.replace("active","");}//Showthecurrenttab,andaddan"active"classtothebuttonthatopenedthetabdocument.getElementById(cityName).style.display="block";evt.currentTarget.className+="active";}'
