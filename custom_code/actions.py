from __future__ import absolute_import, division, print_function

import datetime
import json
import logging
import time
from typing import Text, List, Dict, Any

import pandas as pd
from rasa_core_sdk import Action, Tracker, ActionExecutionRejection
from rasa_core_sdk.events import (
    SlotSet,
    Restarted,
    Form,
    ActionReverted,  # reverts before bot's action
    UserUtteranceReverted,  # reverts before user's uttered
    FollowupAction,
)
from rasa_core_sdk.executor import CollectingDispatcher
from rasa_core_sdk.forms import FormAction, REQUESTED_SLOT
from tabulate import tabulate

logger = logging.getLogger(__name__)

NOW = datetime.datetime.now()

# time to keeping product
PRODUCT_KEPT_TIME = 3000

# TODO: replace all mappings with ml model(s)
# intents mapping
vi_intents_mapping = json.load(open("custom_code/vi_intents.json", "r"))

# packages mapping
packages_mapping = json.load(open("custom_code/package_mapping.json", "r"))

# scopes mapping
scopes_mapping = json.load(open("custom_code/scope_mapping.json", "r"))

# entities mapping
product_mapping = json.load(open("custom_code/product_mapping.json", "r"))

for k, v in scopes_mapping.items():
    for x in v:
        product_mapping[x] = k

# product dataframe
productdf = pd.read_csv("custom_code/data.csv")

# current products (sender_id as key)
current_products = {}


def cvt_number(num):
    return "{0:,}".format(num).replace(",", ".")


def intent_ranking_tabular(tracker: Tracker):
    try:
        logger.debug(
            "message: {}\n".format(tracker.latest_message["text"])
            + tabulate(
                [
                    [intent["name"], intent["confidence"]]
                    for intent in tracker.latest_message["intent_ranking"]
                ],
                headers=["intent", "confidence"],
                tablefmt="orgtbl",
            )
        )

    except Exception as e:
        logger.error("No intent on {}".format(tracker.latest_message["text"]))


class product_form(FormAction):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    @staticmethod
    def required_slots(tracker):  # type: (Tracker) -> List[Text]
        return ["pname", "org_field", "ppack", "pscopes"]

    def slot_mappings(self):  # type: () -> Dict[Text: Union[Dict, List[Dict]]]
        return {
            x: self.from_entity(
                x,
                intent=[
                    "inform",
                    "change_product",
                    "buy",
                    "ask_price",
                    "ask_extended_price",
                    "ask_trial",
                    "ask_sales",
                    "ask_training",
                    "complain_price",
                ],
            )
            for x in ["pname", "org_field", "ppack"]
        }

    def request_next_slot(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type: Dict[Text, Any]
    ):  # type: (...) -> Optional[List[Dict]]
        for slot in self.required_slots(tracker):
            if (
                slot == "pscopes"
                and tracker.get_slot("ppack") is not None
                and not tracker.get_slot("called_pscopes")
            ):
                ppack = tracker.get_slot("ppack")
                pscopes = tracker.get_slot("pscopes")

                if pscopes == "16" or ppack == "enterprise":
                    # pscopes may be None
                    return None

                if pscopes == "11" or ppack == "standard":
                    dispatcher.utter_template("utter_ask_pscopes_11", tracker)

                elif pscopes == "13" or ppack == "professional":
                    dispatcher.utter_template("utter_ask_pscopes_13", tracker)

                return [SlotSet(REQUESTED_SLOT, "pscopes")]

            elif tracker.get_slot(slot) is None:
                logger.debug("Request next slot '{}'".format(slot))
                dispatcher.utter_template(
                    "utter_ask_{}".format(slot),
                    tracker,
                    silent_fail=False,
                    **tracker.slots
                )
                return [SlotSet(REQUESTED_SLOT, slot)]

        logger.debug("No slots left to request")
        return None

    def validate(
        self, dispatcher, tracker, domain
    ):  # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
        slot_values = self.extract_other_slots(dispatcher, tracker, domain)
        new_slot_values = {}

        slot_to_fill = tracker.get_slot(REQUESTED_SLOT)
        if slot_to_fill:
            slot_values.update(self.extract_requested_slot(dispatcher, tracker, domain))

            if not slot_values:
                raise ActionExecutionRejection(
                    self.name(),
                    "Failed to validate slot {0} "
                    "with action {1}"
                    "".format(slot_to_fill, self.name()),
                )

        for slot, value in slot_values.items():
            logger.debug("slot: {} - value: {}".format(slot, value))

            if type(value) == str:
                value = value.strip().lower()

            if slot == "pname":
                pvalue = product_mapping.get(value, value)

                # confusing case (sme)
                # if value == "kế_toán":
                #     dispatcher.utter_template("utter_ask_pname_slt", tracker)
                #     slot_values[slot] = None
                #
                # else:
                if pvalue not in ["sme"]:
                    dispatcher.utter_template("utter_ask_pname", tracker)
                    slot_values[slot] = None

                else:
                    slot_values[slot] = pvalue  # sme

                    # predict pscopes and ppack if value is scope's name
                    try:
                        svalue = scopes_mapping[pvalue][value]

                        if svalue is not None:  # ["11", "13", "16"]
                            r = productdf.loc[
                                (productdf["pname"] == pvalue)
                                & (productdf["pscopes"] >= float(svalue))
                            ]
                            if len(r) > 0:
                                new_slot_values["ppack"] = r.iloc[0]["ppack"]

                        new_slot_values["pscopes"] = svalue

                    except Exception as e:
                        pass

            elif slot == "org_field":
                pvalue = packages_mapping.get(value, value)

                if pvalue not in ["standard", "professional", "enterprise"]:
                    dispatcher.utter_template("utter_ask_org_field", tracker)
                    slot_values[slot] = None

                else:
                    # try to set values for ppack and pscopes
                    new_slot_values["ppack"] = pvalue

                    # required pname
                    pname = tracker.get_slot("pname")
                    if pname is None:
                        pname = slot_values.get("pname", None)

                    if pname is not None:
                        r = productdf.loc[
                            (productdf["pname"] == pname)
                            & (productdf["ppack"] == pvalue)
                        ]
                        if len(r) > 0:
                            new_slot_values["pscopes"] = str(r.iloc[0]["pscopes"])

            elif slot == "ppack":
                value = packages_mapping.get(value, value)

                if value not in ["standard", "professional", "enterprise"]:
                    dispatcher.utter_template("utter_ask_org_field", tracker)
                    slot_values[slot] = None

                else:
                    # try to set values for ppack and pscopes
                    new_slot_values[slot] = value

                    # required pname
                    pname = tracker.get_slot("pname")
                    if pname is not None:
                        r = productdf.loc[
                            (productdf["pname"] == pname)
                            & (productdf["ppack"] == value)
                        ]
                        if len(r) > 0:
                            new_slot_values["pscopes"] = str(r.iloc[0]["pscopes"])

            elif slot == "pscopes":
                new_slot_values["called_pscopes"] = True

                r = productdf.loc[productdf["pscopes"] >= float(value)]
                if len(r) > 0:
                    new_slot_values["ppack"] = r.iloc[0]["ppack"]

            slot_values = {**slot_values, **new_slot_values}
            # skip asking pscopes ("16")
            if (
                slot_values.get("pscopes", None) == "16"
                or slot_values.get("ppack", None) == "enterprise"
            ):
                slot_values["pscopes"] = "16"
                slot_values["ppack"] = "enterprise"
                slot_values["called_pscopes"] = True

        return [SlotSet(slot, value) for slot, value in slot_values.items()]

    def submit(
        self, dispatcher, tracker, domain
    ):  # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
        pname = tracker.get_slot("pname")
        ppack = tracker.get_slot("ppack")
        org_field = tracker.get_slot("org_field")

        try:
            pscopes = float(tracker.get_slot("pscopes"))

        except Exception as e:
            pscopes = 0  # ???

        r = productdf.loc[
            (productdf["pname"] == pname)
            & (productdf["ppack"] == ppack)
            & (productdf["pscopes"] >= pscopes)
        ]

        if len(r) == 0:
            r = productdf.loc[
                (productdf["pname"] == pname) & (productdf["pscopes"] >= pscopes)
            ]

        if len(r) > 0:
            r = r.iloc[0]

            # TODO: show prices with intents
            # if tracker.latest_message["intent"]["name"] in [
            #     "change_product",
            #     "buy",
            #     "ask_price",
            # ]:
            dispatcher.utter_message(
                "Em tư vấn cho anh/chị gói {} đáp ứng tốt nghiệp vụ kế toán của doanh nghiệp {}. "
                "Gói sản phẩm {} {} có giá là {} VNĐ."
                "".format(
                    r["ppack"].upper(),
                    org_field.replace("_", " "),
                    pname.upper(),
                    r["ppack"].upper(),
                    cvt_number(r["pprice"]),
                )
            )

            # save product to df
            current_products[tracker.sender_id] = {
                "pname": r["pname"],
                "org_field": org_field,
                "ppack": r["ppack"],
                "pscopes": pscopes,
                "pprice": r["pprice"],
                "puprice": r["puprice"],
                "timestamp": time.time(),
            }

        else:
            # deadend
            dispatcher.utter_message("Không có sản phẩm phù hợp với yêu cầu")

        return [SlotSet("called_pscopes", False)] + [
            SlotSet(x, None) for x in self.required_slots(tracker)
        ]

    def run(
        self, dispatcher, tracker, domain
    ):  # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
        # activate the form
        events = self._activate_if_required(tracker)
        # validate user input
        events.extend(self._validate_if_required(dispatcher, tracker, domain))

        # check that the form wasn't deactivated in validation
        if Form(None) not in events:

            # create temp tracker with populated slots from `validate` method
            temp_tracker = tracker.copy()
            for e in events:
                if e["event"] == "slot":
                    temp_tracker.slots[e["name"]] = e["value"]

            next_slot_events = self.request_next_slot(dispatcher, temp_tracker, domain)

            if next_slot_events is not None:
                # request next slot
                events.extend(next_slot_events)
            else:
                # there is nothing more to request, so we can submit
                events.extend(self.submit(dispatcher, temp_tracker, domain))
                # deactivate the form after submission
                events.extend(self.deactivate())

                events.append(FollowupAction("action_listen"))

        return events


class action_change_product(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        curr = current_products.get(tracker.sender_id, None)
        logger.debug(curr)
        if curr is None or time.time() - curr["timestamp"] >= PRODUCT_KEPT_TIME:
            # remove product
            try:
                current_products.pop(tracker.sender_id)

            except Exception as e:
                pass

            dispatcher.utter_message(
                "Anh/chị chưa chọn sản phẩm nào, mời anh/chị chọn sản phẩm"
            )

            return [ActionReverted(), Form("product_form")]  # neednt callback

        else:
            # ERROR: update not replace
            # current_products.pop(tracker.sender_id)

            dispatcher.utter_message(
                "Anh/chị đang đổi sản phẩm {} {}".format(
                    curr["pname"].upper(), curr["ppack"].upper()
                )
            )

            # set org_field, ppack, pscopes to slots
            return [
                ActionReverted(),
                SlotSet("org_field", curr["org_field"]),
                SlotSet("ppack", curr["ppack"]),
                SlotSet("pscopes", curr["pscopes"]),
                Form("product_form"),
            ]


class action_buy(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        curr = current_products.get(tracker.sender_id, None)
        logger.debug(curr)
        if curr is None or time.time() - curr["timestamp"] >= PRODUCT_KEPT_TIME:
            # remove product
            try:
                current_products.pop(tracker.sender_id)

            except Exception as e:
                pass

            # dispatcher.utter_message(
            #     "Anh/chị chưa chọn sản phẩm nào, mời anh/chị chọn sản phẩm"
            # )

            return [ActionReverted(), Form("product_form")]  # neednt callback

        dispatcher.utter_message(
            "Anh/chị đang mua sản phẩm {} {} với giá {} VNĐ\n"
            "Anh/chị truy cập vào link này để thực hiện mua phần mềm và thanh toán đơn hàng ạ:\n"
            "http://www.misa.com.vn/san-pham/mua-hang?Step=3&PurchaseProduct=MISASME2019"
            "".format(
                curr["pname"].upper(), curr["ppack"].upper(), cvt_number(curr["pprice"])
            )
        )

        return []


class action_extended_price_response(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        curr = current_products.get(tracker.sender_id, None)
        logger.debug(curr)
        if curr is None or time.time() - curr["timestamp"] >= PRODUCT_KEPT_TIME:
            # remove product
            try:
                current_products.pop(tracker.sender_id)

            except Exception as e:
                pass

            dispatcher.utter_message(
                "Anh/chị chưa chọn sản phẩm nào, mời anh/chị chọn sản phẩm"
            )

            # callback to response
            return [ActionReverted(), Form("product_form")]  # callback

        dispatcher.utter_message(
            "Hàng năm anh/chị sẽ trả thêm phí cập nhật những thay đổi về chế độ kế toán, "
            "chính sách thuế và các tính năng, tiện ích mới của sản phẩm. "
            "Phí cập nhật hàng năm của gói {} là {} đồng/năm"
            "".format(curr["ppack"].upper(), cvt_number(curr["puprice"]))
        )

        return []


class action_trial_response(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        curr = current_products.get(tracker.sender_id, None)
        logger.debug(curr)
        if curr is None or time.time() - curr["timestamp"] >= PRODUCT_KEPT_TIME:
            # remove product
            try:
                current_products.pop(tracker.sender_id)

            except Exception as e:
                pass

            dispatcher.utter_message(
                "Anh/chị chưa chọn sản phẩm nào, mời anh/chị chọn sản phẩm"
            )

            # callback to response
            return [ActionReverted(), Form("product_form")]  # callback

        dispatcher.utter_message(
            "Hiện tại, với sản phẩm {} {}, MISA cho khách hàng 15 ngày dùng thử "
            "sản phẩm để trải nghiệm trước khi quyết định mua."
            "Anh/chị truy cập vào link này điền thông tin đăng ký dùng thử và tải bộ cài nhé:"
            "http://www.misa.com.vn/san-pham/download/pid/158/MISA-SMENET-2019"
            "".format(curr["pname"].upper(), curr["ppack"].upper())
        )

        return []


class action_sales_response(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        curr = current_products.get(tracker.sender_id, None)
        logger.debug(curr)
        if curr is None or time.time() - curr["timestamp"] >= PRODUCT_KEPT_TIME:
            # remove product
            try:
                current_products.pop(tracker.sender_id)

            except Exception as e:
                pass

            dispatcher.utter_message(
                "Anh/chị chưa chọn sản phẩm nào, mời anh/chị chọn sản phẩm"
            )

            # callback to response
            return [ActionReverted(), Form("product_form")]  # callback

        if NOW.month == 4 and NOW.year == 2019:  # query in sales_db
            dispatcher.utter_message(
                "Trong thời gian từ 1/4/2019-30/4/2019, sản phẩm {} có chương trình khuyến mại sau:\n"
                "1. Tặng voucher trị giá 2.950.000 đ cho doanh nghiệp mới thành lập 3 tháng cuối năm 2018 và 2019\n"
                "2. Tặng 500 hóa đơn khi mua MISA SME.NET 2019 kèm dịch vụ hóa đơn điện tử meInvoice.vn\n"
                "3. Tặng voucher 2.000.000 đ cho doanh nghiệp nâng cấp từ gói Starter lên gói cao hơn"
                "".format(curr["pname"].upper())
            )

        else:
            dispatcher.utter_message(
                "Hiện tại, sản phẩm {} không có khuyến mãi nào".format(
                    curr["pname"].upper()
                )
            )

        return []


class action_ask_training(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        curr = current_products.get(tracker.sender_id, None)
        logger.debug(curr)
        if curr is None or time.time() - curr["timestamp"] >= PRODUCT_KEPT_TIME:
            # remove product
            try:
                current_products.pop(tracker.sender_id)

            except Exception as e:
                pass

            dispatcher.utter_message(
                "Anh/chị chưa chọn sản phẩm nào, mời anh/chị chọn sản phẩm"
            )

            # callback to response
            return [ActionReverted(), Form("product_form")]  # callback

        dispatcher.utter_button_message(
            "Với phần mềm {} {}, "
            "MISA cung cấp dịch vụ đào tạo hướng dẫn sử dụng trong 2 ngày gồm hình thức"
            "".format(curr["pname"].upper(), curr["ppack"].upper()),
            buttons=[
                {
                    "title": "Đào tạo tập trung",
                    "payload": '/inform{"training": "centralized"}',
                },
                {
                    "title": "Đào tạo trực tiếp tại doanh nghiệp",
                    "payload": '/inform{"training": "onsite"}',
                },
                {
                    "title": "Hỗ trợ triển khai phần mềm",
                    "payload": '/inform{"training": "deployment"}',
                },
            ],
        )

        return []


class action_training_response(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        training = tracker.get_slot("training")

        if training == "centralized":
            dispatcher.utter_message(
                "Đào tạo hướng dẫn sử dụng phần "
                "mềm trong khóa tập huấn tập trung "
                "do MISA tổ chức trong 02 ngày tại "
                "Hà Nội, Hồ Chí Minh."
            )

        elif training == "onsite":
            dispatcher.utter_message(
                "Đào tạo hướng dẫn sử dụng phần "
                "mềm trực tiếp tại đơn vị trong 2 "
                "ngày cho dưới 10 cán bộ áp dụng "
                "đối với khách hàng tại Hà Nội, Hồ "
                "Chí Minh. Khách hàng ở ngoài địa "
                "điểm trên phải trả thêm phí đi lại, "
                "ăn nghỉ cho cán bộ MISA. "
                "Khách hàng chịu trách nhiệm "
                "chuẩn bị máy móc, thiết bị phục vụ "
                "cho việc đào tạo"
            )

        elif training == "deployment":
            dispatcher.utter_message(
                "Mỗi khóa chỉ tư vấn triển khai cho "
                "một nghiệp vụ như: Kế toán, Bán "
                "hàng, Nhân sự, Quản trị chung.Đây "
                "là mức giá tối thiểu được thực hiện "
                "trong thời gian tối đa là 5 ngày. "
                "Trường hợp nếu khách hàng muốn "
                "thêm thời gian tư vấn thì gói dịch "
                "vụ tư vấn triển khai bổ sung là "
                "5.000.000đ/ngày."
            )

        return []


class action_complain_price_response(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        curr = current_products.get(tracker.sender_id, None)
        logger.debug(curr)
        if curr is None or time.time() - curr["timestamp"] >= PRODUCT_KEPT_TIME:
            # remove product
            try:
                current_products.pop(tracker.sender_id)

            except Exception as e:
                pass

            dispatcher.utter_message(
                "Anh/chị chưa chọn sản phẩm nào, mời anh/chị chọn sản phẩm"
            )

            # callback to response
            return [ActionReverted(), Form("product_form")]  # callback

        dispatcher.utter_message(
            "Với sản phẩm {} {} có giá {} VNĐ và phí cập nhật hàng năm là {} VNĐ "
            "anh/chị sẽ có được:\n"
            "- Chất lượng sản phẩm cao: đầy đủ nghiệp vụ, tính năng thông minh, báo cáo điều "
            "hành quản trị nhiều tiêu chí, phát triển sản phẩm theo tiêu chuẩn quốc tế (CMMI level 3)\n"
            "- Dịch vụ hỗ trợ tốt: Hơn 100 nhân viên tư vấn 24/7, 365 ngày/năm, đa dạng kênh hỗ trợ\n"
            "- Luôn cập nhật kịp thời các thay đổi của chính sách nhà nước\n"
            "- Sản phẩm SME kết nối trực tiếp với cơ quan thuế để kê khai thuê qua mạng, "
            "phát hành hóa đơn điện tử trực tiếp trên PM, kết nối NH điện tử."
            "".format(
                curr["pname"].upper(),
                curr["ppack"].upper(),
                cvt_number(curr["pprice"]),
                cvt_number(curr["puprice"]),
            )
        )

        return []


class action_reset(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]

        dispatcher.utter_template("utter_thanks", tracker)

        try:
            # remove current product
            current_products.pop(tracker.sender_id)

        except Exception as e:
            pass

        return [Restarted()]


# two-stage fallback actions
class action_default_ask_affirmation(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        intent_ranking_tabular(tracker)

        # TODO: fix loop on this fallback action
        last_intent = tracker.latest_message["intent"]["name"]

        entities = {x["entity"]: x["value"] for x in tracker.latest_message["entities"]}
        if len(entities) > 0:
            affirm_payload = "/{}{}".format(last_intent, entities)

        else:
            affirm_payload = "/{}".format(last_intent)

        dispatcher.utter_button_message(
            "Có phải anh/chị muốn {}".format(
                vi_intents_mapping.get(last_intent, last_intent)
            ),
            buttons=[
                {"title": "Đúng", "payload": affirm_payload},
                {"title": "Sai", "payload": "/out_of_scope"},
            ],
        )

        # TODO: break twostagefallback, not call rephrase
        return [UserUtteranceReverted(), ActionReverted()]


class action_default_ask_rephrase(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        dispatcher.utter_template("utter_fallback", tracker)

        return []


# fallback action
class action_default_fallback(Action):
    def name(self):  # type: () -> Text
        return self.__class__.__name__

    def run(
        self,
        dispatcher,  # type: CollectingDispatcher
        tracker,  # type: Tracker
        domain,  # type:  Dict[Text, Any]
    ):  # type: (...) -> List[Dict[Text, Any]]
        dispatcher.utter_message(
            "Hiện tại câu hỏi của anh/chị em không trả lời được. "
            "Em sẽ liên hệ với nhân viên tư vấn để giải đáp thắc mắc của anh/chị. "
            "Cảm ơn anh/chị"
        )

        return [UserUtteranceReverted(), ActionReverted()]
