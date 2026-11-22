# 해피워크 앱 아이콘 (D안)

해피워크 아이콘은 흰 바탕에 해피에이징 로고를 원래 색 그대로 두고, 오른쪽 아래에 하늘색(`#1E9DF1`, 앱의 primary 색) 발자국 배지를 붙인 모양이다. 회사 로고는 손대지 않고 배지로 '걷기 앱'이라는 차이를 준다. 로고는 네 고리 조각이 정사각형 배치에서 왼쪽으로 약 18° 기울어진 원본 형태이며, 조각이 겹친 곳은 곱하기 합성으로 진하게 보인다. 이름 옆에는 「with 해피에이징」을 함께 적어 회사와의 관계를 밝힌다.

모든 파일은 `render-icons.mjs` 하나로 만든다. 모양이나 색을 바꿀 때는 스크립트를 고친 뒤 저장소 루트에서 다시 실행한다(외부 패키지 필요 없음).

```powershell
node mobile/brand/render-icons.mjs
```

## 이 폴더의 원본

| 파일 | 쓰임 |
|---|---|
| `happywalk-icon.svg`, `happywalk-icon-1024.png` | 기본 아이콘. 흰 둥근 사각형 바탕, 바깥은 투명. 스토어·문서용 큰 이미지 |
| `happywalk-icon-maskable.svg`, `happywalk-icon-maskable-1024.png` | 흰 꽉 찬 사각형 바탕. 기기가 원·둥근 사각 등으로 잘라도 배지까지 남도록 내용을 폭의 62%로 줄임 |
| `happywalk-icon-foreground.svg`, `happywalk-icon-foreground-432.png` | 안드로이드 적응형 아이콘의 전경(로고와 배지만, 바탕 투명, 내용 폭 50%). 배경은 흰색 |

배지 속 그림은 신발 자국 두 개다. 자국 하나는 앞창(타원 두 개)과 조금 떨어진 반원 뒤축이며, 두 자국을 엇갈려 왼쪽 위(로고 쪽)로 걸어가게 놓는다. 앱 화면의 로고(`src/ui/components/brand.tsx`의 `FootBadge`)도 같은 값을 쓴다(스크립트의 `SHOE`).

## PWA가 쓰는 파일(`mobile/public/`)

`icon-192.png`, `icon-512.png`(기본), `icon-maskable-512.png`(마스크용), `apple-touch-icon.png`(아이폰 홈 화면, 180px), `favicon.svg`(브라우저 탭). 연결은 `index.html`과 `manifest.webmanifest`에 있다. manifest의 `background_color`는 흰색이라, 홈 화면에서 열 때 OS가 보여 주는 첫 화면과 앱의 시작 화면(`src/ui/components/Splash.tsx`)이 같은 흰 바탕으로 이어진다.

## 아직 하지 않은 것

안드로이드 앱(`mobile/android`)의 실행 아이콘(`mipmap-*`)에는 아직 적용하지 않았다. 적용할 때는 위 전경 파일과 흰 배경으로 적응형 아이콘을 만든다.
