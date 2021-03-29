


function changeImage(picture_id, path) {
    alert("hiiiiiii");
    // document.getElementById("big").src = "images/" + path + ".PNG"
    // document.getElementById(string(picture_pos)).src = "images/" + path + ".PNG"
    document.getElementById(picture_id).src = "images/" + path + ".PNG"
    }

// function changeImage1() {
//     alert("hi");
//     document.getElementById("big").src = "images/test01.PNG"
// }
//     function changeImage2() {
//     document.getElementById("big").src = "images/test02.PNG"
// }
//     function changeImage3() {
//     document.getElementById("big").src = "images/test03.PNG"
// }



// var bigPic = document.querySelector("#big"); // 큰사진
// var smallPics = document.querySelectorAll(".small");
// var audio = document.querySelectorAll(".demo");

// for(var i=0; i<audio.length; i++){
//     audio[i].addEventListener("click", changepic); // 이벤트 처리 // 처음에 한번만 실행되고,,, 막상 audio를 실행하면 안되네?
//     // bigPic.setAttribute("src", "images/test03.png");
// }

// function changePic(idx) {
//     if (idx == 1) {
//         bigPic.setAttribute("src", "images/test01.png");
//     }
//     if (idx == 2) {
//         bigPic.setAttribute("src", "images/test02.png");
//     }
//     if (idx == 3) {
//         bigPic.setAttribute("src", "images/test03.png");
//     }
// }

// function changepic(){
//     // var smallPicAttribute = this.getAttribute("id"); // 이건 audio의 src네.. picture의 src로 바꿔줘야한다.\
//     bigPic.setAttribute("src", "images/test03.png");
//     var smallPicAttribute = this.getAttribute("id"); // getElementById
    
//     if (smallPicAttribute == "target1") {
//         bigPic.setAttribute("src", "images/test01.png");
//     }
//     if (smallPicAttribute == "target2") {
//         bigPic.setAttribute("src", "images/test02.png");
//     }
//     if (smallPicAttribute == "target3") {
//         bigPic.setAttribute("src", "images/test03.png");
//     }
    
//     // bigPic.setAttribute("src", "images/test03.png");
//     // bigPic.setAttribute("src", smallPicAttribute);
// }


// ref: https://pridiot.tistory.com/162




// for(var i=0; i<smallPics.length; i++){
//     smallPics[i].addEventListener("click", changepic); // 이벤트 처리
// }

// function changepic(){
//     var smallPicAttribute = this.getAttribute("src");
//     bigPic.setAttribute("src", smallPicAttribute);
// }

