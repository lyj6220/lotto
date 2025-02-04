from main import app, db
from models import Category, Item

def init_database():
    with app.app_context():
        # 데이터베이스 테이블 생성
        db.create_all()
        
        # 메인 카테고리 생성
        food = Category(name="음식")
        hobby = Category(name="취미 및 레저")
        etc = Category(name="기타")
        db.session.add_all([food, hobby, etc])
        db.session.commit()

        # 음식 서브카테고리
        food_subcategories = {
            "한식": ["동태찌개", "사태찌개", "닭도리탕", "삼계탕", "백숙", "백반", "한정식", "국밥", "감자탕", "족발", "보쌈", "닭발", "닭똥집", "꼬막무침", "골뱅이무침", "번데기탕", "홍어", "육회"],
            "중식": ["짜장면", "짬뽕", "마라탕", "훠궈", "양꼬치", "깐풍기", "탕수육", "마라샹궈", "어향동고", "깐쇼새우", "동파육", "꿔바로우"],
            "일식": ["회", "초밥", "라멘", "돈카츠", "이자카야", "우동", "타코야끼", "오코노미야끼", "사시미", "가라아게", "나가사키 짬뽕"],
            "양식": ["스테이크", "파스타", "피자", "치킨", "햄버거", "감바스", "하몽", "나초", "치즈 플래터", "바베큐립"],
            "동남아 음식": ["쌀국수", "팟타이", "카레", "난", "탄두리치킨", "분짜", "똠얌꿍", "바인미", "뿌팟퐁커리"],
            "분식": ["떡볶이", "김밥", "치킨", "햄버거", "핫도그", "순대", "오징어튀김", "야끼만두", "어묵탕"],
            "고기구이": ["삼겹살", "소고기구이", "닭갈비", "양고기구이", "조개구이", "조개찜", "장어구이", "닭꼬치", "돼지막창", "곱창구이", "대창구이", "석화구이"],
            "주류": ["소주", "맥주", "사케", "보드카", "위스키", "와인", "막걸리", "청주", "고량주", "전통약주", "테킬라"]
        }

        for subcat_name, items in food_subcategories.items():
            subcat = Category(name=subcat_name, parent_id=food.id)
            db.session.add(subcat)
            db.session.commit()
            
            for item_name in items:
                item = Item(name=item_name, category_id=subcat.id)
                db.session.add(item)
        
        # 취미 서브카테고리
        hobby_subcategories = {
            "실내 취미": ["스크린골프", "노래연습장", "게임", "헬스", "크로스핏", "당구", "포켓볼", "볼링", "요가", "필라테스", "다트", "스포츠 슈팅", "보드게임카페", "방탈출카페", "만화카페", "VR게임", "사격장"],
            "아웃도어 레저": ["트레킹", "등산", "캠핑", "자전거", "마라톤", "스포츠 클라이밍", "국궁", "승마", "오프로드 바이크", "롱보드", "스케이트보드", "골프", "낚시", "루지", "ATV", "승마체험", "번지점프"],
            "수상 스포츠": ["서핑", "스쿠버다이빙", "스노클링", "웨이크보드", "워터스키", "제트스키", "카약", "카누", "래프팅", "바나나보트", "플라이보드", "수영", "패들보드", "요트", "스킨스쿠버", "낚시배투어"],
            "항공 스포츠": ["패러글라이딩", "행글라이딩", "스카이다이빙", "열기구 체험", "스키 패러글라이딩", "경비행기 체험"],
            "설상 & 빙상 스포츠": ["스키", "스노보드", "스노모빌", "빙벽 등반", "스노슈잉", "아이스 스케이팅", "아이스하키", "컬링", "스피드 스케이팅", "얼음낚시"]
        }

        for subcat_name, items in hobby_subcategories.items():
            subcat = Category(name=subcat_name, parent_id=hobby.id)
            db.session.add(subcat)
            db.session.commit()
            
            for item_name in items:
                item = Item(name=item_name, category_id=subcat.id)
                db.session.add(item)

        # 기타 서브카테고리
        etc_subcategories = {
            "유흥 & 모임": ["시내 술집", "국일관", "노상", "편의점", "놀이공원", "포장마차", "실내포차", "감성주점", "와인바", "칵테일바", "클럽"],
            "여행 & 숙박": ["1박 2일 여행", "캠핑카 여행", "글램핑", "펜션 여행", "한옥스테이"]
        }

        for subcat_name, items in etc_subcategories.items():
            subcat = Category(name=subcat_name, parent_id=etc.id)
            db.session.add(subcat)
            db.session.commit()
            
            for item_name in items:
                item = Item(name=item_name, category_id=subcat.id)
                db.session.add(item)

        db.session.commit()

if __name__ == "__main__":
    init_database() 