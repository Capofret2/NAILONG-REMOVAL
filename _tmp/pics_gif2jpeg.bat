cd ./pics
powershell -Command "Get-ChildItem *.gif | ForEach-Object {
    $output = $_.BaseName + '.jpeg'
    ffmpeg -i $_.FullName -vf 'select=eq(n,0)' -vframes 1 $output
}"
powershell -Command "Remove-Item *.gif"
