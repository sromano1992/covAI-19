import React from 'react';
import { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/css/bootstrap.css';
import Form from 'react-bootstrap/Form'
import Button from 'react-bootstrap/Button'
import Table from 'react-bootstrap/Table'
import Col from 'react-bootstrap/Col'
import Row from 'react-bootstrap/Row'
import Container from 'react-bootstrap/Container'
import Card from 'react-bootstrap/Card'
import Figure from 'react-bootstrap/Figure'
import Alert from 'react-bootstrap/Alert'
import $ from 'jquery';

class App extends React.Component {

  constructor(props) {
    super(props);
    this.amount = React.createRef();
    this.getCharts()
  }

  getCharts = () => {
    //ITALY CHART
    $.ajax({
      crossDomain: true,
      type: 'POST',
      //url: 'http://localhost:8000/predict',
      url: 'http://35.238.217.244:8000/predict',
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Cache-Control": "no-cache",
        "Postman-Token": "ed61f085-da3f-4768-bd25-bd94618a5eca"
      },
      data: {
        "country": "Italy"
      },
      success: function(data) { 
        console.log('backend response received!'); 
        console.log(data);
        var obj = JSON.parse(data)
        //$('#italy_charts').append(obj['chart']);
        $('#italy_img').attr("src", "data:image/upng;base64," + obj['chart'])
      },
      error: function() { alert('Failed!'); }
    })

    //CHINA CHART
    $.ajax({
      crossDomain: true,
      type: 'POST',
      //url: 'http://localhost:8000/predict',
      url: 'http://35.238.217.244:8000/predict',
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Cache-Control": "no-cache",
        "Postman-Token": "ed61f085-da3f-4768-bd25-bd94618a5eca"
      },
      data: {
        "country": "China"
      },
      success: function(data) { 
        console.log('backend response received!'); 
        console.log(data);
        var obj = JSON.parse(data)
        //$('#italy_charts').append(obj['chart']);
        $('#china_img').attr("src", "data:image/png;base64," + obj['chart'])
      },
      error: function() { alert('Failed!'); }
    })
    
  };

  render(){
    return (
      <div className="App">



        <Table striped bordered hover>
  <thead>
    <tr>
      <td colSpan="2">
              <h1>covAI-19: Covid-19 diffusion prediction in Italy</h1>
              Welcome to covAI-19 dashboard. <br></br>
              This webpage will report a prediction about the number of Confirmed, Deaths, and Recovered cases in the next days in Italy and China.
The number you see are generated using a machine learning model trained with the Covid-19 China evolution: the assumption is that if the model will be able to predict the China evolution on the historical data, we can predict the Italy (and also other countries) evolution using the same trained model.
<br></br>
            <h2>References</h2>
            <b>Author: </b> Simone Romano - <a href="https://www.linkedin.com/in/simoneromano92/">linkedIn profile</a><br></br>
            <b>Articles:</b> <a href="https://medium.com/@s.romano1992/covai-19-covid-19-diffusion-prediction-in-italy-part-1-e7d20f653b46">Part 1</a> - <a href="https://medium.com/@s.romano1992/covai-19-covid-19-diffusion-prediction-in-italy-part-2-132c72d4947c">Part 2</a> - <a href="https://medium.com/@s.romano1992/covai-19-covid-19-diffusion-prediction-in-italy-part-3-d524f6e37ca9">Part 3</a>
            
            <h2>Developers</h2>
            A REST service is available to get the predictions for Italy and China for the next week. An example of interaction via curl cli:<br></br>

            <i>curl -X POST http://35.238.217.244:8000/predict -H 'Accept: application/json' -H 'Cache-Control: no-cache' -H 'Content-Type: application/x-www-form-urlencoded' -d country=Italy</i>
      </td>
    </tr>
  </thead>
  <tbody>
    <tr>
    <td>
              <h1>Italy predictions</h1>
              <div id='italy_charts'>
                <img id='italy_img' src="" width="500px" heigth="500px"></img>
              </div>
            </td>
    </tr>
    <tr>
            <td>
              <h1>China predictions</h1>
              <div id='china_chart'>
                <img id='china_img' src="" width="500px" heigth="500px"></img>
              </div>
            </td>
    </tr>
  </tbody>
</Table>
      </div>
    );
  }
}

export default App;
