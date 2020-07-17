var MyModule;
$('#toggle').click(function() {
   $(this).toggleClass('active');
  console.log($('#toggle').attr('class'));
   $('#overlay').toggleClass('open');
});