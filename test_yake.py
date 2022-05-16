import yake

text = "重庆晚报6月11日报道 昨日，市政府公众信息网发布消息称，经2010年5月13日市政府第70次常务会议通过，给予文强、陈洪刚二人行政开除处分。今年4月14日，市第五中级人民法院以受贿罪，包庇、纵容黑社会性质组织罪，巨额财产来源不明罪，强奸罪数罪并罚判处文强死刑，剥夺政治权利终身，并处没收个人全部财产。5月21日，市高级人民法院对文强案二审宣判，依法驳回文强上诉，维持一审的死刑判决。2月25日，市公安局交警总队原总队长陈洪刚受贿案在市第五中级人民法院一审宣判。陈洪刚因犯受贿，包庇、纵容黑社会性质组织，巨额财产来源不明，伪造居民身份证罪，数罪并罚，被判处有期徒刑20年，没收个人财产40万元人民币，追缴赃款326万余元及不明来源财产584万余元。"

kw_extractor = yake.KeywordExtractor(lan="zh",top=5)
keywords = kw_extractor.extract_keywords(text)

for kw in keywords:
    print(kw)