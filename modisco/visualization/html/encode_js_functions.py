#helpers for using Python to generate JavaScript functions
#Note: for faster loading in the browser, all extraneous characters (i.e. space, tab, newline) removed


def showHist():
    return 'function showHist(hist_div){\n\
    var connected_div=document.getElementById(hist_div).style.display;\n\
    var vis="block";\n\
    if(connected_div=="none"){\n\
    vis="block";\n\
    }\n\
    if(connected_div=="block"){\n\
    vis="none";\n\
    }\n\
    document.getElementById(hist_div).style.display=vis;}'


def accordion():
    return 'var acc=document.getElementsByClassName("accordion");\n\
    var i;\n\
    for(i=0;i<acc.length;i++){\n\
    acc[i].onclick=function(){\n\
    this.classList.toggle("active");\n\
    var panel=this.nextElementSibling;\n\
    if(panel.style.maxHeight){\n\
    panel.style.maxHeight=null;}\n\
    else{\n\
    panel.style.maxHeight=panel.scrollHeight+"px";}}}\n'

def def_modalFunction():
    return 'function getModal(modalId,imageId,modalImageId,captionId,closeId){\n\
    var modal=document.getElementById(modalId);\n\
    var img=document.getElementById(imageId).childNodes[0].childNodes[0];\n\
    var modalImg=document.getElementById(modalImageId);\n\
    var captionText=document.getElementById(captionId);\n\
    img.onclick=function(){\n\
    modal.style.display="block";\n\
    modalImg.src=this.src;}\n\
    var span=document.getElementById(closeId);\n\
    span.onclick=function(){\n\
    modal.style.display="none";}}\n'

def call_modalFunction(modalId,imageId,modalImageId,captionId,closeId):
    return 'getModal("{modalId}","{imageId}","{modalImageId}","{captionId}","{closeId}");\n'.format(modalId=modalId,
                                                                                    imageId=imageId,
                                                                                    modalImageId=modalImageId,
                                                                                    captionId=captionId,
                                                                                    closeId=closeId)

def uncheckAll():
    return 'var checkboxes=document.getElementsByTagName("input");\n\
    for(var i=0; i < checkboxes.length; i++){\n\
    if(checkboxes[i].type=="checkbox"){\n\
    checkboxes[i].checked=false;}}'


def selectClusterTab():
    return 'function openCluster(evt,cityName){\n\
    var i,tabcontent,tablinks;\n\
    tabcontent=document.getElementsByClassName("tabcontent");\n\
    for(i=0;i<tabcontent.length;i++){\n\
    tabcontent[i].style.display="none";}\n\
    tablinks=document.getElementsByClassName("tablinks");\n\
    for(i=0;i<tablinks.length;i++){\n\
    tablinks[i].className=tablinks[i].className.replace("active","");}\n\
    document.getElementById(cityName).style.display="block";\n\
    evt.currentTarget.className+="active";}'
