$(document).ready(function(){
    $('.bxslider').bxSlider({
      mode: 'horizontal',  // 슬라이드 모드를 수평(horizontal)으로 설정합니다.
      speed: 500,          // 슬라이드 전환 속도를 설정합니다.
      pause: 5000,         // 자동 재생 시 한 장의 이미지가 보여지는 시간을 설정합니다.
      auto: true,          // 자동 재생 여부를 설정합니다.
      pager: true,         // 페이저를 보여줄지 여부를 설정합니다.
      controls: false,     // 이전/다음 버튼을 보여줄지 여부를 설정합니다.
      autoControls: true,  // 자동 재생 시작/정지 버튼을 보여줄지 여부를 설정합니다.
      slideMargin: 10,     // 슬라이드 간격을 설정합니다.
      minSlides: 2,        // 슬라이드가 최소 몇 장 이상 보여져야 하는지를 설정합니다.
      maxSlides: 5,        // 슬라이드가 최대 몇 장까지 보여질 수 있는지를 설정합니다.
      slideWidth: 170,     // 슬라이드의 너비를 설정합니다.
      slideHeight: 170,    // 슬라이드의 높이를 설정합니다.
    });
  });