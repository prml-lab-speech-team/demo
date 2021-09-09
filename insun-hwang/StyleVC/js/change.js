


function changeImage(picture_id, path) {
    // alert("images/plot/" + path + ".PNG");
    document.getElementById(picture_id).src = "images/plot/" + path + ".PNG"
}


// ref: https://pridiot.tistory.com/162

// for(var i=0; i<smallPics.length; i++){
//     smallPics[i].addEventListener("click", changepic); // 이벤트 처리
// }

// function changepic(){
//     var smallPicAttribute = this.getAttribute("src");
//     bigPic.setAttribute("src", smallPicAttribute);
// }

