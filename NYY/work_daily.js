function addRow() {
    // table element 찾기
    const table = document.getElementById('fruits');
    
    // 새 행(Row) 추가
    const newRow = table.insertRow();
    
    // 새 행(Row)에 Cell 추가
    const newCell1 = newRow.insertCell(0);
    const newCell2 = newRow.insertCell(1);
    const newCell3 = newRow.insertCell(2);
    const newCell4 = newRow.insertCell(3);
    
    // Cell에 텍스트 추가
    newCell1.innerText = '';
    newCell2.innerText = '';
    newCell3.innerText = '';
    newCell4.innerText = '';
  }