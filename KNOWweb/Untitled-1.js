// create an html file that has a button on it. When the button is clicked it should prompt the user to enter text.var button = document.getElementById('button');

button.onclick = function(){
    var text = prompt('Enter Text');
    console.log(text);
  }