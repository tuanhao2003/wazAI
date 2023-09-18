let slides= document.getElementsByClassName("slide");
let currentIndex=0;
let scrollable = true;

function reLoad() {
    while(currentIndex>0){
        prevSlide();
        currentIndex--;
    }
    window.scrollTo(0, 0);
}

function nextSlide(){
    if(currentIndex < slides.length-1) {
        slides[currentIndex].style.transform=`translateY(-${(currentIndex+1)*100}%)`;
        slides[currentIndex].style.transition="1s ease-in-out";
        currentIndex+=1;
        slides[currentIndex].style.transform=`translateY(-${currentIndex*100}%)`;
        slides[currentIndex].style.transition="1s ease-in-out";
    }
    else{
        currentIndex=currentIndex;
    }
}

function prevSlide(){
    if(currentIndex > 0) {
        slides[currentIndex].style.transform="translateY(0%)";
        slides[currentIndex].style.transition="1s ease-in-out";
        currentIndex-=1;
        slides[currentIndex].style.transform=`translateY(-${(currentIndex)*100}%)`;
        slides[currentIndex].style.transition="1s ease-in-out";
    }
    else{
        currentIndex=currentIndex;
    }
}


document.addEventListener("DOMContentLoaded", () => {
    document.addEventListener("wheel", (event)=> {
        if (scrollable==true) {
            if(event.deltaY > 25){
                nextSlide();
                scrollable=false;
                setTimeout(()=>{
                    scrollable=true;
                }, 1500);
            }
            else if(event.deltaY < -25){
                prevSlide();
                scrollable=false;
                setTimeout(()=>{
                    scrollable=true;
                }, 1500);
            }
        }
    });
});

function unCover(cover){
    if(cover.onclick && !cover.classList.contains('unCover')){
        cover.classList.add('unCover');
        document.getElementById('laThao').classList.add('moveOn');
        document.getElementById('klTrai1').classList.add('active');
        document.getElementById('klTrai2').classList.add('active');
        document.getElementById('klPhai1').classList.add('active');
        document.getElementById('klPhai2').classList.add('active');
    }
    else if(cover.onclick && cover.classList.contains('unCover')){
        cover.classList.remove('unCover');
        document.getElementById('laThao').classList.remove('moveOn');
        document.getElementById('klTrai1').classList.remove('active');
        document.getElementById('klTrai2').classList.remove('active');
        document.getElementById('klPhai1').classList.remove('active');
        document.getElementById('klPhai2').classList.remove('active');
    }
}

function beeUp(){
    if(!document.getElementById('bee').classList.contains('fly')){
        document.getElementById('bee').classList.add('fly');
        setTimeout(() => {
            document.getElementById('bee').classList.remove('fly');
        }, 10000);
    }
    else{
        document.getElementById('bee').classList.remove('fly');
        setTimeout(() => {
            document.getElementById('bee').classList.add('fly');
        }, 500);
    }
}

function klMove(){
    if(!document.getElementById('happy').classList.contains('active')){
        document.getElementById('happy').classList.add('active');
        setTimeout(() => {
            document.getElementById('happy').classList.remove('active');
        }, 10000);
    }
    else{
        document.getElementById('happy').classList.remove('active');
        setTimeout(() => {
            document.getElementById('happy').classList.add('active');
        }, 500);
    }
}

function showImg(){
    if(!document.getElementById('hiddenImg').classList.contains('active')){
        document.getElementById('hiddenImg').classList.add('active');
        document.getElementById('hiddenTitle').classList.add('active');
    }
}

function cmsn(){
    if(!document.getElementById('tt4').classList.contains('active')){
        document.getElementById('tt4').classList.add('active');
        document.getElementById('ct4').classList.add('active');
    }
    else{
        document.getElementById('tt4').classList.remove('active');
        document.getElementById('ct4').classList.remove('active');
    }
}