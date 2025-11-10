说明/例子

| ea | 等于= | 例：eg("name" "老王”) →name=’老王” |
| ne | 不等于 | 例：ne("name", "老王”) →name>’老王’ |
| gt | 大于> | 例： gt("age", 18) ->age〉18 |
| ge | 大于等于= | 例： ge("age", 18) →age = 18 |
| lt | 小干〈 | 例： lt("age",18) →age<18 |
| le | 小于= | 例： le("age", 18) →age=18 |
| between | BETWEEN 值1AND值2 | 例： between("age", 18,30) -->age between 18 and3 |
| notBetween | NOT BETWEEN 值1AD值2 | 例： notBetween("age", 18, 30) ->age not between 1 |
| like | LIKE '%值%’ | 例： like("name" 王 >name like '%王% |
| notLike | NOT LIKE'%值%’ | 例：not.ike("name "王") >name not like '%T% |
| likeLeft | LIKE'%值” | 例： likeLeft("name" 王”) >name like '%王 |
| likeRight | LIKE'值% | 例：likeRight("name" 王") >name like '王% |
| isNu11 | 字段ISNULL | 例：isNu1l("name >name is null |
| isNotNu11 | 字段ISNOT NULL | 例：isNotNull("name") -→name is not null |
| in | 字段 IN(vO.V1.) | 例·in("age".{1.2.3}) ---age in (1.23 |
| notIn | 字段 NOT IN (vO, v1, ..) | 例:notIn("age", 1, 2, 3) ->age not in (1,2,3) |
| inSql | 字段IN(sql语句) | inSql("id", "select id from table where id < 3") ->id in (select id from table where id <3)  |
| groupBy | 分组：GROUP BY字段,. | 例:groupBy("id", "name")- -->group by id,name |
| orderByAsc | 排序:ORDER BY 字段,...ASC | 例:orderByAsc("id", name")--->order by id ASC,nam |
| vLclDyisY ArderRwDasn | J:IwLLLIXI···IW 排序ORDEERV字段DESC | v·Lelpyioe\ iuylldlleirolucl y lt ip, llcll 例l.ArderRyDecc("id" "nam")---Sorder hv id DESC n |
| orderBy | 排序:ORDER BY 字段, .. | 例:orderBy(true, true, "id", "name") -->order by id ASC,name ASC  |
| having | HAVING（sql语句） | having("sum(age) > {O}", 11)--->having sum(age） > 1 |
| o | 拼接OR | 注意事项： 主动调用or表示紧接着下一个方法不是用and连接!(不调用用and连接) 例:eq("id",1).or().eq("name","老王"）--->id = 1 or n |
| and | AND 嵌套 | 例:and(i -> i.eq("name",“李白").ne("status",“活着 --→and（name ='李白’and status<>’活着’）  |
| apply | 拼接sql | 注意事项： 该方法可用于数据库函数动态入参的params对应前面salH {index}部分.这样是不会有sql注入风险的,反之会有! 例:apply("date format(dateColumn, '%Y-%m-%d')= {O} 08")--->date format(dateColumn,'%Y-%m-%d'） = '2008- |
| last | 无视优化规则直接拼接到sql的最后 | 无视优化规则直接拼接到sq_的最后 注意事项： 只能调用一次，多次调用以最后一次为准有sq1注入的风险例:last("limit 1")  |
| exists | 拼接EXISTS(sql语句) | 例: exists("select id from table where age = 1) >exists (select id from table where age= l) |

l8 and 30

ne ASC ame DESC

lor则默认为使 $\mathrm{\ z a m e} ~=~^{\prime} {\frac{\#} {\Xi}} \pm$ 着")）

laving内部的 ."，“2008-08- $\scriptstyle-0 8-0 8^{\prime} \,^{\prime\prime} )$ 

 $\grave{\iota},$ 请谨慎使用

notExists 拼接NOT EXISTS（sql语句） 例: notExists("select id from table where age - 1) $= 1 )$ 
->not exists (select id from table where age
正常嵌套不带AND或者OR
nested 正常嵌套不带AND或者OR 例：nested $( {\dot{1}} \ -\gg{\dot{1}}$ eaCgnamcq “ 皇自健3上peCSRtBs Ttuos
-(name ='李白’and status "活着’)blo sdn. net,
