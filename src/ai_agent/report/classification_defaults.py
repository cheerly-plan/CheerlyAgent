# 自动生成，请勿手工编辑：分类规则的只读降级快照。
# 由 scripts/sync_indicators_snapshot.py 导出；主配置不可用时回退到这里。

COMPANY_TYPES = frozenset(
    {
        'bank',
        'cyclical',
        'equipment',
        'fabless',
        'foundry',
        'generic',
        'industry_shared',
        'manufacturing',
        'packaging',
        'real_estate',
        'robotics',
        'semiconductor',
        'software',
    }
)

TYPE_ALIASES = {
    'bank': 'bank',
    '银行': 'bank',
    'generic': 'generic',
    'manufacturing': 'manufacturing',
    '一般制造': 'manufacturing',
    '制造业': 'manufacturing',
    'semiconductor': 'semiconductor',
    '半导体': 'semiconductor',
    'robotics': 'robotics',
    '机器人': 'robotics',
    'fabless': 'fabless',
    '设计': 'fabless',
    'foundry': 'foundry',
    '制造': 'foundry',
    'equipment': 'equipment',
    '设备': 'equipment',
    'packaging': 'packaging',
    '封测': 'packaging',
    'software': 'software',
    '软件': 'software',
    'real_estate': 'real_estate',
    '地产': 'real_estate',
    'cyclical': 'cyclical',
    '周期': 'cyclical',
    'industry_shared': 'industry_shared',
}

NAME_HINTS = (
    ('银行', 'bank'),
    ('置业', 'real_estate'),
    ('地产', 'real_estate'),
    ('房地产', 'real_estate'),
    ('软件', 'software'),
    ('互联', 'software'),
    ('云', 'software'),
    ('钢铁', 'cyclical'),
    ('化工', 'cyclical'),
    ('煤炭', 'cyclical'),
    ('有色', 'cyclical'),
    ('机器人', 'robotics'),
    ('谐波', 'manufacturing'),
    ('减速器', 'manufacturing'),
)

SEGMENT_TO_TYPE = (
    ('设计', 'fabless'),
    ('制造', 'foundry'),
    ('设备', 'equipment'),
    ('封测', 'packaging'),
    ('本体', 'manufacturing'),
    ('减速器', 'manufacturing'),
    ('伺服控制', 'equipment'),
    ('系统集成', 'manufacturing'),
)

INDUSTRY_SEGMENTS = {
    'bank': {'国有大行': 'bank', '股份制': 'bank', '城商行': 'bank'},
    'semiconductor': {'设计': 'fabless', '制造': 'foundry', '设备': 'equipment', '封测': 'packaging'},
    'robotics': {'伺服控制': 'equipment', '本体': 'manufacturing', '减速器': 'manufacturing', '系统集成': 'software'},
}

INDUSTRY_LEAD = {
    'bank': {'lead': ['营业收入', '归属于母公司所有者的净利润', '净利润', '净息差', '不良贷款率', '不良贷款拨备覆盖率'], 'lead_hint': ['净息差', '不良贷款率', '不良贷款拨备覆盖率', '资本充足率']},
}
