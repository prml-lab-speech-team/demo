


function changeImage(picture_id, path) {
    // alert("images/plot/" + path + ".PNG");
    // document.getElementById(picture_id).src = "images/plot/" + path + ".PNG"
    // alert("wav/melspec/" + path + ".PNG");
    // alert("wav/melsepc/" + path + ".PNG");
    // document.getElementById(picture_id).src = 'wav/melsepc/' + path + '.PNG'
    document.getElementById(picture_id).src = path

    // alert("wav/melsepc/" + path + ".PNG"); # 'wav/melsepc/p269_p269_001_to_p256_p256_001.PNG'
    // document.getElementById(picture_id).src = 'wav/melspec/p269_p269_001_to_p256_p256_001.PNG'
    
}


// ref: https://pridiot.tistory.com/162

// for(var i=0; i<smallPics.length; i++){
//     smallPics[i].addEventListener("click", changepic); // 이벤트 처리
// }

// function changepic(){
//     var smallPicAttribute = this.getAttribute("src");
//     bigPic.setAttribute("src", smallPicAttribute);
// }

