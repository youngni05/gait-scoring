param([string]$TargetUrl,[string]$OutputName,[long]$Start=-1,[long]$End=-1,[long]$MaximumBytes=5242880)
$ErrorActionPreference='Stop'
$s=Split-Path -Parent $MyInvocation.MyCommand.Path
$request=[System.Net.HttpWebRequest]::Create($TargetUrl)
$request.Timeout=30000
$request.ReadWriteTimeout=30000
if ($Start -ge 0) { $request.AddRange($Start,$End) }
$response=$request.GetResponse()
try {
 $status=[int]$response.StatusCode
 if ($Start -ge 0 -and $status -ne 206) { throw "Partial response required; got $status; body was not read" }
 if ($response.ContentLength -gt $MaximumBytes) { throw 'Content-Length exceeded cap; body was not read' }
 $stream=$response.GetResponseStream()
 $memory=New-Object System.IO.MemoryStream
 try {
  $buffer=New-Object byte[] 16384
  while (($n=$stream.Read($buffer,0,$buffer.Length)) -gt 0) {
   if ($memory.Length+$n -gt $MaximumBytes) { throw 'Stream exceeded cap' }
   $memory.Write($buffer,0,$n)
  }
  $data=$memory.ToArray()
 } finally { $stream.Close(); $memory.Dispose() }
 if ($Start -ge 0 -and $data.Length -ne ($End-$Start+1)) { throw 'Range byte count mismatch' }
 [System.IO.File]::WriteAllBytes((Join-Path $s $OutputName),$data)
 $result=@{url=$TargetUrl;output=$OutputName;status=$status;bytes=$data.Length;content_range=$response.Headers['Content-Range'];content_type=$response.ContentType;sha256=(Get-FileHash -LiteralPath (Join-Path $s $OutputName) -Algorithm SHA256).Hash.ToLower();max_bytes=$MaximumBytes;start=$Start;end=$End;at=[DateTime]::UtcNow.ToString('o')}
 $result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $s ($OutputName+'.http.json')) -Encoding UTF8
 $result | ConvertTo-Json -Compress
} finally { $response.Close() }
